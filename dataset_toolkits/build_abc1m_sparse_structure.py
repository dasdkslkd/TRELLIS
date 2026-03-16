"""
Build a TRELLIS-ready 80-channel sparse-structure dataset from ABC-1M parquet shards.

Channel layout matches the Fusion360 80-channel convention:
    [0]     occupancy
    [1:8]   one-hot BRep class with order:
                0 Boundary
                1 BSpline surface
                2 Plane
                3 Torus
                4 Cone
                5 Sphere
                6 Cylinder
    [8:80]  24 sampled surface points per occupied voxel (24 x 3)

Design choices for stability:
    - Non-analytic OCCT face types are merged into BSpline surface.
    - Boundary voxels are marked from ABC edge polyline samples, not mesh adjacency.
    - Surface occupancy is built from face grid points plus midpoints/quad centers.
    - Per-voxel sample points are chosen deterministically with FPS-style subsampling.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


CLASS_ORDER = [
    "BOUNDARY",
    "BSPLINE_SURFACE",
    "PLANE",
    "TORUS",
    "CONE",
    "SPHERE",
    "CYLINDER",
]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
ANALYTIC_FACE_MAP = {
    0: "PLANE",
    1: "CYLINDER",
    2: "CONE",
    3: "SPHERE",
    4: "TORUS",
}
REQUIRED_COLUMNS = [
    "stem",
    "face_points_normalized",
    "face_mask",
    "face_types",
    "edge_points_normalized",
]


@dataclass
class BuildStats:
    seen: int = 0
    success: int = 0
    skipped_existing: int = 0
    skipped_invalid: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ABC-1M TRELLIS sparse structure dataset")
    parser.add_argument(
        "--abc-root",
        type=str,
        default="/public/home/pb22000140/xjn/datasets/ABC-1M",
        help="ABC-1M parquet root",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_trellis_split",
        help="Output dataset root",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n-sample-points", type=int, default=24)
    parser.add_argument("--target-count", type=int, default=100000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8, help="Parquet row batch size")
    parser.add_argument("--max-parquets", type=int, default=0, help="Limit number of parquet files for smoke tests")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="Optional path for manifest json; defaults to <output-root>/manifest.json",
    )
    return parser.parse_args()


def decode_npy_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def map_face_type(code: int) -> str:
    return ANALYTIC_FACE_MAP.get(int(code), "BSPLINE_SURFACE")


def voxel_size(resolution: int) -> float:
    return 2.0 / float(resolution)


def points_to_flat_indices(points: np.ndarray, resolution: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    scaled = np.floor((pts + 1.0) * 0.5 * resolution).astype(np.int64)
    scaled = np.clip(scaled, 0, resolution - 1)
    return scaled[:, 0] * resolution * resolution + scaled[:, 1] * resolution + scaled[:, 2]


def flat_to_coords(flat_indices: np.ndarray, resolution: int) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    i = flat_indices // (resolution * resolution)
    rem = flat_indices % (resolution * resolution)
    j = rem // resolution
    k = rem % resolution
    return np.stack([i, j, k], axis=1)


def append_grouped_points(
    store: Dict[int, Dict[str, List[np.ndarray]]],
    points: np.ndarray,
    class_name: str,
    resolution: int,
) -> None:
    if points.size == 0:
        return
    flats = points_to_flat_indices(points, resolution)
    order = np.argsort(flats, kind="stable")
    flats = flats[order]
    points = points[order]
    unique_flats, starts = np.unique(flats, return_index=True)
    ends = list(starts[1:]) + [len(flats)]
    for flat, start, end in zip(unique_flats.tolist(), starts.tolist(), ends):
        bucket = store.setdefault(flat, {})
        bucket.setdefault(class_name, []).append(points[start:end])


def build_face_point_store(
    face_points: np.ndarray,
    face_mask: np.ndarray,
    face_types: np.ndarray,
    resolution: int,
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    voxel_points: Dict[int, Dict[str, List[np.ndarray]]] = {}
    num_faces = int(face_points.shape[0])
    for face_idx in range(num_faces):
        class_name = map_face_type(int(face_types[face_idx]))
        pts = np.asarray(face_points[face_idx], dtype=np.float32)
        mask = np.asarray(face_mask[face_idx], dtype=bool)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if not mask.any():
            continue

        append_grouped_points(voxel_points, pts[mask], class_name, resolution)

        full_quads = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
        if np.any(full_quads):
            quad_centers = (
                pts[:-1, :-1] + pts[1:, :-1] + pts[:-1, 1:] + pts[1:, 1:]
            ) * 0.25
            append_grouped_points(voxel_points, quad_centers[full_quads], class_name, resolution)

        vertical_pairs = mask[:-1, :] & mask[1:, :]
        if np.any(vertical_pairs):
            vertical_mid = (pts[:-1, :] + pts[1:, :]) * 0.5
            append_grouped_points(voxel_points, vertical_mid[vertical_pairs], class_name, resolution)

        horizontal_pairs = mask[:, :-1] & mask[:, 1:]
        if np.any(horizontal_pairs):
            horizontal_mid = (pts[:, :-1] + pts[:, 1:]) * 0.5
            append_grouped_points(voxel_points, horizontal_mid[horizontal_pairs], class_name, resolution)

    return voxel_points


def build_boundary_voxels(edge_points: np.ndarray, resolution: int) -> set[int]:
    boundary_flats: set[int] = set()
    if edge_points.size == 0:
        return boundary_flats

    vs = voxel_size(resolution)
    step_size = max(vs * 0.5, 1e-4)
    for polyline in np.asarray(edge_points, dtype=np.float32):
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline[:-1], polyline[1:]):
            if not np.isfinite(start).all() or not np.isfinite(end).all():
                continue
            seg_len = float(np.linalg.norm(end - start))
            steps = max(2, int(math.ceil(seg_len / step_size)) + 1)
            ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
            pts = start[None, :] * (1.0 - ts) + end[None, :] * ts
            boundary_flats.update(points_to_flat_indices(pts, resolution).tolist())
    return boundary_flats


def choose_class(class_points: Dict[str, List[np.ndarray]]) -> str:
    best_class = None
    best_count = -1
    for class_name in CLASS_ORDER[1:]:
        arrays = class_points.get(class_name)
        count = 0 if arrays is None else int(sum(arr.shape[0] for arr in arrays))
        if count > best_count:
            best_class = class_name
            best_count = count
    return best_class or "BSPLINE_SURFACE"


def deduplicate_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points

    # Quantized hash dedup is much cheaper than np.unique(axis=0) on large voxel-local point sets.
    quantized = np.round(points * 1e5).astype(np.int32)
    seen = set()
    kept = []
    for idx, item in enumerate(quantized):
        key = (int(item[0]), int(item[1]), int(item[2]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(idx)
    return points[np.asarray(kept, dtype=np.int64)]


def cap_candidate_points(points: np.ndarray, max_candidates: int = 512) -> np.ndarray:
    if len(points) <= max_candidates:
        return points

    centroid = points.mean(axis=0, keepdims=True)
    dist = np.sum((points - centroid) ** 2, axis=1)
    order = np.argsort(dist, kind="stable")
    if max_candidates == 1:
        return points[order[:1]]

    anchors = np.linspace(0, len(order) - 1, max_candidates, dtype=np.int64)
    return points[order[anchors]]


def deterministic_fps(points: np.ndarray, target_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = cap_candidate_points(points)
    points = deduplicate_points(points)
    if len(points) == 0:
        return np.zeros((target_count, 3), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, target_count, axis=0)

    centroid = points.mean(axis=0, keepdims=True)
    dist_to_center = np.sum((points - centroid) ** 2, axis=1)
    selected = [int(np.argmin(dist_to_center))]
    min_dist = np.sum((points - points[selected[0]]) ** 2, axis=1)

    while len(selected) < min(target_count, len(points)):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dist = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    sampled = points[selected]
    if len(sampled) < target_count:
        reps = int(math.ceil(target_count / len(sampled)))
        sampled = np.tile(sampled, (reps, 1))[:target_count]
    return sampled.astype(np.float32)


def build_sparse_tensor(
    face_points: np.ndarray,
    face_mask: np.ndarray,
    face_types: np.ndarray,
    edge_points: np.ndarray,
    resolution: int,
    n_sample_points: int,
) -> torch.Tensor | None:
    voxel_point_store = build_face_point_store(face_points, face_mask, face_types, resolution)
    if not voxel_point_store:
        return None

    boundary_flats = build_boundary_voxels(edge_points, resolution)
    occupied_flats = sorted(voxel_point_store.keys())
    n_channels = 8 + n_sample_points * 3
    coords = flat_to_coords(np.asarray(occupied_flats, dtype=np.int64), resolution).T
    feats = np.zeros((len(occupied_flats), n_channels), dtype=np.float32)
    feats[:, 0] = 1.0

    for row_idx, flat in enumerate(occupied_flats):
        class_points = voxel_point_store[flat]
        if flat in boundary_flats:
            class_name = "BOUNDARY"
            sample_source = np.concatenate([arr for arrays in class_points.values() for arr in arrays], axis=0)
        else:
            class_name = choose_class(class_points)
            sample_arrays = class_points.get(class_name, [])
            if sample_arrays:
                sample_source = np.concatenate(sample_arrays, axis=0)
            else:
                sample_source = np.concatenate([arr for arrays in class_points.values() for arr in arrays], axis=0)

        feats[row_idx, 1 + CLASS_TO_INDEX[class_name]] = 1.0
        sampled = deterministic_fps(sample_source, n_sample_points)
        feats[row_idx, 8:8 + n_sample_points * 3] = sampled.reshape(-1)

    sparse = torch.sparse_coo_tensor(
        torch.as_tensor(coords, dtype=torch.long),
        torch.as_tensor(feats, dtype=torch.float32),
        (resolution, resolution, resolution, n_channels),
    )
    return sparse.coalesce()


def iter_parquet_paths(abc_root: str) -> List[str]:
    root = Path(abc_root)
    parquet_paths = sorted(str(path) for path in root.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {abc_root}")
    return parquet_paths


def iter_rows(parquet_paths: Iterable[str], batch_size: int) -> Iterator[Tuple[str, dict]]:
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=REQUIRED_COLUMNS):
            frame = batch.to_pydict()
            batch_len = len(frame["stem"])
            for idx in range(batch_len):
                row = {key: frame[key][idx] for key in REQUIRED_COLUMNS}
                yield parquet_path, row


def ensure_split_dirs(output_root: str) -> Dict[str, Path]:
    root = Path(output_root)
    train_dir = root / "train"
    val_dir = root / "val"
    for split_dir in [train_dir, val_dir]:
        (split_dir / "data").mkdir(parents=True, exist_ok=True)
    return {"train": train_dir, "val": val_dir}


def count_existing(split_dir: Path) -> int:
    data_dir = split_dir / "data"
    return sum(1 for _ in data_dir.glob("*.pt"))


def append_metadata_row(metadata_path: Path, stem: str) -> None:
    header = not metadata_path.exists()
    with metadata_path.open("a", encoding="utf-8") as f:
        if header:
            f.write("sha256\n")
        f.write(f"{stem}\n")


def append_record(record_path: Path, record: dict) -> None:
    with record_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def choose_split(train_count: int, val_count: int, train_target: int, val_target: int) -> str | None:
    if train_count < train_target:
        return "train"
    if val_count < val_target:
        return "val"
    return None


def write_manifest(manifest_path: Path, args: argparse.Namespace) -> None:
    manifest = {
        "source": os.path.abspath(args.abc_root),
        "output_root": os.path.abspath(args.output_root),
        "resolution": args.resolution,
        "n_sample_points": args.n_sample_points,
        "target_count": args.target_count,
        "val_count": args.val_count,
        "train_count": args.target_count - args.val_count,
        "class_order": CLASS_ORDER,
        "analytic_face_map": ANALYTIC_FACE_MAP,
        "non_analytic_policy": "all non-analytic OCCT surface types -> BSPLINE_SURFACE",
        "boundary_policy": "edge_points_normalized rasterized to voxel paths, overriding surface class",
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def build_dataset(args: argparse.Namespace) -> BuildStats:
    if args.val_count >= args.target_count:
        raise ValueError("val-count must be smaller than target-count")

    split_dirs = ensure_split_dirs(args.output_root)
    manifest_path = Path(args.manifest_path) if args.manifest_path else Path(args.output_root) / "manifest.json"
    records_path = Path(args.output_root) / "build_records.jsonl"
    write_manifest(manifest_path, args)

    train_target = args.target_count - args.val_count
    val_target = args.val_count

    train_count = count_existing(split_dirs["train"])
    val_count = count_existing(split_dirs["val"])

    stats = BuildStats(success=train_count + val_count)
    parquet_paths = iter_parquet_paths(args.abc_root)
    if args.max_parquets > 0:
        parquet_paths = parquet_paths[:args.max_parquets]

    for parquet_path, row in iter_rows(parquet_paths, args.batch_size):
        if train_count >= train_target and val_count >= val_target:
            break

        stats.seen += 1
        stem = str(row["stem"])
        existing_train = split_dirs["train"] / "data" / f"{stem}.pt"
        existing_val = split_dirs["val"] / "data" / f"{stem}.pt"
        if existing_train.exists() or existing_val.exists():
            stats.skipped_existing += 1
            continue

        try:
            face_points = decode_npy_blob(row["face_points_normalized"])
            face_mask = decode_npy_blob(row["face_mask"])
            face_types = decode_npy_blob(row["face_types"])
            edge_points = decode_npy_blob(row["edge_points_normalized"])
        except Exception as exc:
            stats.skipped_invalid += 1
            append_record(records_path, {
                "stem": stem,
                "status": "invalid_blob",
                "error": str(exc),
                "parquet": parquet_path,
            })
            continue

        try:
            sparse = build_sparse_tensor(
                face_points=face_points,
                face_mask=face_mask,
                face_types=face_types,
                edge_points=edge_points,
                resolution=args.resolution,
                n_sample_points=args.n_sample_points,
            )
        except Exception as exc:
            stats.skipped_invalid += 1
            append_record(records_path, {
                "stem": stem,
                "status": "build_failed",
                "error": str(exc),
                "parquet": parquet_path,
            })
            continue

        if sparse is None or sparse._nnz() == 0:
            stats.skipped_invalid += 1
            append_record(records_path, {
                "stem": stem,
                "status": "empty",
                "parquet": parquet_path,
            })
            continue

        split = choose_split(train_count, val_count, train_target, val_target)
        if split is None:
            break

        save_path = split_dirs[split] / "data" / f"{stem}.pt"
        torch.save(sparse, save_path)
        append_metadata_row(split_dirs[split] / "metadata.csv", stem)
        append_record(records_path, {
            "stem": stem,
            "status": "success",
            "split": split,
            "parquet": parquet_path,
            "nnz": int(sparse._nnz()),
        })

        stats.success += 1
        if split == "train":
            train_count += 1
        else:
            val_count += 1

        if stats.success % 100 == 0:
            print(
                f"[progress] success={stats.success} train={train_count}/{train_target} "
                f"val={val_count}/{val_target} seen={stats.seen} "
                f"skipped_existing={stats.skipped_existing} skipped_invalid={stats.skipped_invalid}"
            )

    print(
        f"[done] success={stats.success} train={train_count}/{train_target} "
        f"val={val_count}/{val_target} seen={stats.seen} "
        f"skipped_existing={stats.skipped_existing} skipped_invalid={stats.skipped_invalid}"
    )
    return stats


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()