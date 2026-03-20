from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
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
SCAN_COLUMNS = [
    "stem",
    "face_points_normalized",
    "face_mask",
    "face_types",
    "edge_points_normalized",
    "face_bbox_world",
    "edge_bbox_world",
    "face_edge_incidence",
    "scaled_unique",
    "err",
]
PREFILTER_REASON_PREFIX = "skip_"
NEIGHBOR_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def decode_npy_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def map_face_type(code: int) -> str:
    return ANALYTIC_FACE_MAP.get(int(code), "BSPLINE_SURFACE")


def batch_ncs2wcs(points: np.ndarray, bbox_world: np.ndarray) -> np.ndarray:
    bbox_world = np.asarray(bbox_world, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if bbox_world.ndim != 2 or bbox_world.shape[1] != 6:
        raise ValueError(f"Expected bbox shape (N, 6), got {bbox_world.shape}")
    if points.shape[0] != bbox_world.shape[0]:
        raise ValueError(
            f"Point/bbox count mismatch: points={points.shape[0]} bboxes={bbox_world.shape[0]}"
        )

    center = (bbox_world[:, :3] + bbox_world[:, 3:]) / 2.0
    size = np.max(bbox_world[:, 3:] - bbox_world[:, :3], axis=1)
    shape = (bbox_world.shape[0],) + (1,) * (points.ndim - 2) + (1,)
    center_shape = (bbox_world.shape[0],) + (1,) * (points.ndim - 2) + (3,)
    return points * (size.reshape(shape) / 2.0) + center.reshape(center_shape)


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

    step_size = max(voxel_size(resolution) * 0.5, 1e-4)
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
) -> Optional[torch.Tensor]:
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
            sample_source = np.concatenate(
                [arr for arrays in class_points.values() for arr in arrays],
                axis=0,
            )
        else:
            class_name = choose_class(class_points)
            sample_arrays = class_points.get(class_name, [])
            if sample_arrays:
                sample_source = np.concatenate(sample_arrays, axis=0)
            else:
                sample_source = np.concatenate(
                    [arr for arrays in class_points.values() for arr in arrays],
                    axis=0,
                )

        feats[row_idx, 1 + CLASS_TO_INDEX[class_name]] = 1.0
        sampled = deterministic_fps(sample_source, n_sample_points)
        feats[row_idx, 8:8 + n_sample_points * 3] = sampled.reshape(-1)

    sparse = torch.sparse_coo_tensor(
        torch.as_tensor(coords, dtype=torch.long),
        torch.as_tensor(feats, dtype=torch.float32),
        (resolution, resolution, resolution, n_channels),
    )
    return sparse.coalesce()


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def iter_parquet_paths(abc_root: str) -> List[str]:
    root = Path(abc_root)
    parquet_paths = sorted(str(path) for path in root.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {abc_root}")
    return parquet_paths


def build_prefilter_reason(
    row: Dict[str, object],
    bit: int,
    max_edge: int,
) -> Optional[str]:
    tol = 1 / (2 ** (bit - 1))

    if row.get("err"):
        return "dataset_err"
    if "scaled_unique" in row and row["scaled_unique"] is False:
        return "scaled_unique_false"

    face_edge_adj = decode_npy_blob(row["face_edge_incidence"]).astype(bool)
    if len(face_edge_adj) == 0:
        return "empty_face_edge_incidence"
    if face_edge_adj.ndim == 0:
        return "scalar_face_edge_incidence"
    if np.any(np.all(np.logical_not(face_edge_adj), axis=1)):
        return "face_without_edges"
    if np.any(face_edge_adj.sum(axis=0) != 2):
        return "non_manifold_edge_incidence"
    if face_edge_adj.shape[1] > max_edge:
        return "too_many_edges"

    face_bbox = decode_npy_blob(row["face_bbox_world"])
    face_diff = np.abs(face_bbox[:, :3] - face_bbox[:, 3:])
    if np.any(np.all(face_diff < tol, axis=-1)):
        return "degenerate_face_bbox"

    edge_bbox = decode_npy_blob(row["edge_bbox_world"])
    edge_diff = np.abs(edge_bbox[:, :3] - edge_bbox[:, 3:])
    if np.any(np.all(edge_diff < tol, axis=-1)):
        return "degenerate_edge_bbox"

    return None


def validate_voxel_inputs(row: Dict[str, object]) -> Optional[str]:
    try:
        face_ncs = decode_npy_blob(row["face_points_normalized"])
        face_mask = decode_npy_blob(row["face_mask"])
        face_types = decode_npy_blob(row["face_types"])
        edge_ncs = decode_npy_blob(row["edge_points_normalized"])
        face_bbox = decode_npy_blob(row["face_bbox_world"])
        edge_bbox = decode_npy_blob(row["edge_bbox_world"])
        face_edge = decode_npy_blob(row["face_edge_incidence"])
    except Exception:
        return "invalid_blob"

    if face_ncs.ndim != 4:
        return "invalid_face_grid_shape"
    if edge_ncs.ndim != 3:
        return "invalid_edge_grid_shape"
    if face_bbox.ndim != 2 or face_bbox.shape[1] != 6:
        return "invalid_face_bbox_shape"
    if edge_bbox.ndim != 2 or edge_bbox.shape[1] != 6:
        return "invalid_edge_bbox_shape"
    if face_edge.ndim != 2:
        return "invalid_face_edge_shape"
    if face_mask.ndim not in (3, 4):
        return "invalid_face_mask_shape"
    if face_types.ndim != 1:
        return "invalid_face_types_shape"

    num_faces, num_edges = face_edge.shape
    if face_ncs.shape[0] != num_faces:
        return "face_grid_count_mismatch"
    if face_bbox.shape[0] != num_faces:
        return "face_bbox_count_mismatch"
    if edge_ncs.shape[0] != num_edges:
        return "edge_grid_count_mismatch"
    if edge_bbox.shape[0] != num_edges:
        return "edge_bbox_count_mismatch"
    if face_mask.shape[0] != num_faces:
        return "face_mask_count_mismatch"
    if face_types.shape[0] != num_faces:
        return "face_types_count_mismatch"
    if not np.asarray(face_mask, dtype=bool).any():
        return "empty_face_mask"
    return None


def build_occupied_flat_indices_from_row(
    row: Dict[str, object],
    resolution: int,
) -> np.ndarray:
    face_ncs = decode_npy_blob(row["face_points_normalized"])
    face_mask = decode_npy_blob(row["face_mask"])
    face_types = decode_npy_blob(row["face_types"])
    face_bbox = decode_npy_blob(row["face_bbox_world"])

    face_points = batch_ncs2wcs(face_ncs, face_bbox)
    voxel_point_store = build_face_point_store(face_points, face_mask, face_types, resolution)
    if not voxel_point_store:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(voxel_point_store.keys()), dtype=np.int64)


def compute_component_metrics(flat_indices: np.ndarray, resolution: int) -> Dict[str, float]:
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    total_voxels = int(flat_indices.shape[0])
    if total_voxels == 0:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_ratio": 0.0,
            "small_component_ratio": 0.0,
            "total_voxels": 0,
        }

    coords = flat_to_coords(flat_indices, resolution)
    occupied = {tuple(coord) for coord in coords.tolist()}
    visited = set()
    component_sizes: List[int] = []

    for coord in occupied:
        if coord in visited:
            continue
        queue = deque([coord])
        visited.add(coord)
        size = 0
        while queue:
            x, y, z = queue.popleft()
            size += 1
            for dx, dy, dz in NEIGHBOR_OFFSETS_26:
                nxt = (x + dx, y + dy, z + dz)
                if nxt in occupied and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        component_sizes.append(size)

    component_sizes.sort(reverse=True)
    largest_component_size = component_sizes[0]
    largest_component_ratio = largest_component_size / float(total_voxels)
    return {
        "component_count": len(component_sizes),
        "largest_component_size": largest_component_size,
        "largest_component_ratio": largest_component_ratio,
        "small_component_ratio": 1.0 - largest_component_ratio,
        "total_voxels": total_voxels,
    }


def should_run_quality_filter(args: argparse.Namespace) -> bool:
    return (
        args.max_voxel_components > 0
        or args.max_small_component_ratio < 1.0
        or args.low_ratio_component_count_threshold > 0
    )


def build_quality_reason(
    row: Dict[str, object],
    resolution: int,
    max_voxel_components: int,
    max_small_component_ratio: float,
    low_ratio_min: float,
    low_ratio_max: float,
    low_ratio_component_count_threshold: int,
) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    occupied_flats = build_occupied_flat_indices_from_row(row, resolution)
    metrics = compute_component_metrics(occupied_flats, resolution)
    if metrics["total_voxels"] == 0:
        return "empty_voxel_grid", metrics
    if max_voxel_components > 0 and metrics["component_count"] > max_voxel_components:
        return "too_many_voxel_components", metrics
    if (
        low_ratio_component_count_threshold > 0
        and low_ratio_min <= metrics["largest_component_ratio"] <= low_ratio_max
        and metrics["component_count"] >= low_ratio_component_count_threshold
    ):
        return "low_largest_component_ratio", metrics
    if (
        max_small_component_ratio < 1.0
        and metrics["small_component_ratio"] > max_small_component_ratio
    ):
        return "too_many_small_components", metrics
    return None, metrics


def build_sparse_tensor_from_row(
    row: Dict[str, object],
    resolution: int,
    n_sample_points: int,
) -> Optional[torch.Tensor]:
    face_ncs = decode_npy_blob(row["face_points_normalized"])
    face_mask = decode_npy_blob(row["face_mask"])
    face_types = decode_npy_blob(row["face_types"])
    edge_ncs = decode_npy_blob(row["edge_points_normalized"])
    face_bbox = decode_npy_blob(row["face_bbox_world"])
    edge_bbox = decode_npy_blob(row["edge_bbox_world"])

    face_points = batch_ncs2wcs(face_ncs, face_bbox)
    edge_points = batch_ncs2wcs(edge_ncs, edge_bbox)
    return build_sparse_tensor(
        face_points=face_points,
        face_mask=face_mask,
        face_types=face_types,
        edge_points=edge_points,
        resolution=resolution,
        n_sample_points=n_sample_points,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ABC-1M sparse structure dataset with official parsing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan parquet shards and collect eligible candidates")
    scan.add_argument("--abc-root", type=str, default="/public/home/pb22000140/xjn/datasets/ABC-1M")
    scan.add_argument("--scan-root", type=str, required=True)
    scan.add_argument("--num-shards", type=int, required=True)
    scan.add_argument("--shard-id", type=int, required=True)
    scan.add_argument("--batch-size", type=int, default=16)
    scan.add_argument("--bit", type=int, default=10)
    scan.add_argument("--max-edge", type=int, default=1000)
    scan.add_argument("--max-parquets", type=int, default=0)
    scan.add_argument("--quality-resolution", type=int, default=64)
    scan.add_argument("--max-voxel-components", type=int, default=0)
    scan.add_argument("--max-small-component-ratio", type=float, default=1.0)
    scan.add_argument("--low-ratio-min", type=float, default=0.9)
    scan.add_argument("--low-ratio-max", type=float, default=0.95)
    scan.add_argument("--low-ratio-component-count-threshold", type=int, default=16)

    select = subparsers.add_parser("finalize-selection", help="Combine shard scans and choose the build list")
    select.add_argument("--scan-root", type=str, required=True)
    select.add_argument("--selected-manifest", type=str, required=True)
    select.add_argument("--target-count", type=int, default=100000)
    select.add_argument("--reserve-count", type=int, default=10000)

    build = subparsers.add_parser("build", help="Build sparse tensors for selected candidates")
    build.add_argument("--selected-manifest", type=str, required=True)
    build.add_argument("--cache-root", type=str, required=True)
    build.add_argument("--num-shards", type=int, required=True)
    build.add_argument("--shard-id", type=int, required=True)
    build.add_argument("--batch-size", type=int, default=8)
    build.add_argument("--resolution", type=int, default=64)
    build.add_argument("--n-sample-points", type=int, default=24)
    build.add_argument("--bit", type=int, default=10)
    build.add_argument("--max-edge", type=int, default=1000)
    build.add_argument("--max-voxel-components", type=int, default=0)
    build.add_argument("--max-small-component-ratio", type=float, default=1.0)
    build.add_argument("--low-ratio-min", type=float, default=0.9)
    build.add_argument("--low-ratio-max", type=float, default=0.95)
    build.add_argument("--low-ratio-component-count-threshold", type=int, default=16)

    finalize = subparsers.add_parser("finalize-dataset", help="Create the final train/val dataset from cached tensors")
    finalize.add_argument("--selected-manifest", type=str, required=True)
    finalize.add_argument("--cache-root", type=str, required=True)
    finalize.add_argument("--output-root", type=str, required=True)
    finalize.add_argument("--target-count", type=int, default=100000)
    finalize.add_argument("--val-count", type=int, default=2000)
    finalize.add_argument("--resolution", type=int, default=64)
    finalize.add_argument("--n-sample-points", type=int, default=24)
    return parser


def run_scan(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")

    parquet_paths = iter_parquet_paths(args.abc_root)
    if args.max_parquets > 0:
        parquet_paths = parquet_paths[:args.max_parquets]

    scan_root = Path(args.scan_root)
    scan_root.mkdir(parents=True, exist_ok=True)
    records_path = scan_root / f"candidates_shard{args.shard_id:02d}.jsonl"
    summary_path = scan_root / f"summary_shard{args.shard_id:02d}.json"
    if records_path.exists():
        records_path.unlink()

    stats = Counter()
    bad_parquet_records: List[dict] = []
    for parquet_index, parquet_path in enumerate(parquet_paths):
        if parquet_index % args.num_shards != args.shard_id:
            continue

        try:
            parquet = pq.ParquetFile(parquet_path)
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=SCAN_COLUMNS):
                columns = batch.to_pydict()
                batch_len = len(columns["stem"])
                for batch_row in range(batch_len):
                    stats["seen"] += 1
                    row = {name: columns[name][batch_row] for name in SCAN_COLUMNS}
                    reason = build_prefilter_reason(row, bit=args.bit, max_edge=args.max_edge)
                    if reason is not None:
                        stats[PREFILTER_REASON_PREFIX + reason] += 1
                        continue

                    validation_reason = validate_voxel_inputs(row)
                    if validation_reason is not None:
                        stats[PREFILTER_REASON_PREFIX + validation_reason] += 1
                        continue

                    if should_run_quality_filter(args):
                        quality_reason, quality_metrics = build_quality_reason(
                            row,
                            resolution=args.quality_resolution,
                            max_voxel_components=args.max_voxel_components,
                            max_small_component_ratio=args.max_small_component_ratio,
                            low_ratio_min=args.low_ratio_min,
                            low_ratio_max=args.low_ratio_max,
                            low_ratio_component_count_threshold=args.low_ratio_component_count_threshold,
                        )
                        if quality_reason is not None:
                            stats[PREFILTER_REASON_PREFIX + quality_reason] += 1
                            continue

                    record = {
                        "stem": str(row["stem"]),
                        "parquet_path": parquet_path,
                        "parquet_index": parquet_index,
                        "row_index": row_offset + batch_row,
                    }
                    if should_run_quality_filter(args) and quality_metrics is not None:
                        record["quality_metrics"] = quality_metrics
                    write_jsonl(records_path, record)
                    stats["candidate"] += 1

                row_offset += batch_len
        except Exception as exc:
            stats["bad_parquet"] += 1
            bad_parquet_records.append(
                {
                    "parquet_path": parquet_path,
                    "parquet_index": parquet_index,
                    "error": str(exc),
                }
            )
            print(
                f"[scan shard {args.shard_id}] skipping bad parquet index={parquet_index} "
                f"path={parquet_path} error={exc}"
            )
            continue

        print(
            f"[scan shard {args.shard_id}] parquet_index={parquet_index} "
            f"seen={stats['seen']} candidate={stats['candidate']}"
        )

    write_json(
        summary_path,
        {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "abc_root": os.path.abspath(args.abc_root),
            "records_path": str(records_path),
            "stats": dict(stats),
            "bad_parquets": bad_parquet_records,
        },
    )


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_finalize_selection(args: argparse.Namespace) -> None:
    scan_root = Path(args.scan_root)
    selected_manifest = Path(args.selected_manifest)
    candidate_paths = sorted(scan_root.glob("candidates_shard*.jsonl"))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate shard files found under {scan_root}")

    candidates: List[dict] = []
    for path in candidate_paths:
        candidates.extend(load_jsonl(path))

    candidates.sort(key=lambda item: (item["parquet_index"], item["row_index"], item["stem"]))
    needed = args.target_count + args.reserve_count
    if len(candidates) < needed:
        raise ValueError(f"Need at least {needed} candidates, found {len(candidates)}")

    if selected_manifest.exists():
        selected_manifest.unlink()
    for rank, record in enumerate(candidates[:needed]):
        selected_record = dict(record)
        selected_record["rank"] = rank
        write_jsonl(selected_manifest, selected_record)

    write_json(
        selected_manifest.with_suffix(".summary.json"),
        {
            "scan_root": str(scan_root),
            "selected_manifest": str(selected_manifest),
            "candidate_count": len(candidates),
            "selected_count": needed,
            "target_count": args.target_count,
            "reserve_count": args.reserve_count,
        },
    )


def group_manifest_by_parquet(records: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[record["parquet_path"]].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: item["row_index"])
    return grouped


def iter_requested_rows(
    parquet_path: str,
    requested_records: Sequence[dict],
    batch_size: int,
) -> Iterator[Tuple[dict, Dict[str, object]]]:
    parquet = pq.ParquetFile(parquet_path)
    requested_records = sorted(requested_records, key=lambda item: item["row_index"])
    requested_pos = 0
    current_row = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=SCAN_COLUMNS):
        columns = batch.to_pydict()
        batch_len = len(columns["stem"])
        batch_end = current_row + batch_len
        while requested_pos < len(requested_records):
            row_index = requested_records[requested_pos]["row_index"]
            if row_index < current_row:
                requested_pos += 1
                continue
            if row_index >= batch_end:
                break
            offset = row_index - current_row
            row = {name: columns[name][offset] for name in SCAN_COLUMNS}
            yield requested_records[requested_pos], row
            requested_pos += 1
        current_row = batch_end
        if requested_pos >= len(requested_records):
            break


def run_build(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")

    selected_records = load_jsonl(Path(args.selected_manifest))
    shard_records = [record for record in selected_records if record["rank"] % args.num_shards == args.shard_id]
    grouped = group_manifest_by_parquet(shard_records)

    cache_root = Path(args.cache_root)
    data_dir = cache_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    build_records_path = cache_root / f"build_records_shard{args.shard_id:02d}.jsonl"
    summary_path = cache_root / f"build_summary_shard{args.shard_id:02d}.json"
    if build_records_path.exists():
        build_records_path.unlink()

    stats = Counter()
    for parquet_path, records in sorted(grouped.items()):
        for record, row in iter_requested_rows(parquet_path, records, args.batch_size):
            stats["requested"] += 1
            stem = record["stem"]
            cache_path = data_dir / f"{record['rank']:06d}_{stem}.pt"

            reason = build_prefilter_reason(row, bit=args.bit, max_edge=args.max_edge)
            if reason is not None:
                write_jsonl(
                    build_records_path,
                    {
                        "rank": record["rank"],
                        "stem": stem,
                        "status": "skip_prefilter",
                        "reason": reason,
                    },
                )
                stats[PREFILTER_REASON_PREFIX + reason] += 1
                continue

            validation_reason = validate_voxel_inputs(row)
            if validation_reason is not None:
                write_jsonl(
                    build_records_path,
                    {
                        "rank": record["rank"],
                        "stem": stem,
                        "status": "skip_prefilter",
                        "reason": validation_reason,
                    },
                )
                stats[PREFILTER_REASON_PREFIX + validation_reason] += 1
                continue

            if should_run_quality_filter(args):
                quality_reason, quality_metrics = build_quality_reason(
                    row,
                    resolution=args.resolution,
                    max_voxel_components=args.max_voxel_components,
                    max_small_component_ratio=args.max_small_component_ratio,
                    low_ratio_min=args.low_ratio_min,
                    low_ratio_max=args.low_ratio_max,
                    low_ratio_component_count_threshold=args.low_ratio_component_count_threshold,
                )
                if quality_reason is not None:
                    write_jsonl(
                        build_records_path,
                        {
                            "rank": record["rank"],
                            "stem": stem,
                            "status": "skip_quality",
                            "reason": quality_reason,
                            "quality_metrics": quality_metrics,
                        },
                    )
                    stats[PREFILTER_REASON_PREFIX + quality_reason] += 1
                    continue

            if cache_path.exists():
                cached_record = {
                    "rank": record["rank"],
                    "stem": stem,
                    "status": "cached",
                    "cache_path": str(cache_path),
                }
                if should_run_quality_filter(args) and quality_metrics is not None:
                    cached_record["quality_metrics"] = quality_metrics
                write_jsonl(build_records_path, cached_record)
                stats["cached"] += 1
                continue

            try:
                sparse = build_sparse_tensor_from_row(
                    row,
                    resolution=args.resolution,
                    n_sample_points=args.n_sample_points,
                )
            except Exception as exc:
                write_jsonl(
                    build_records_path,
                    {
                        "rank": record["rank"],
                        "stem": stem,
                        "status": "build_failed",
                        "error": str(exc),
                    },
                )
                stats["build_failed"] += 1
                continue

            if sparse is None or sparse._nnz() == 0:
                write_jsonl(
                    build_records_path,
                    {
                        "rank": record["rank"],
                        "stem": stem,
                        "status": "empty",
                    },
                )
                stats["empty"] += 1
                continue

            torch.save(sparse, cache_path)
            write_jsonl(
                build_records_path,
                {
                    "rank": record["rank"],
                    "stem": stem,
                    "status": "success",
                    "cache_path": str(cache_path),
                    "nnz": int(sparse._nnz()),
                    **(
                        {"quality_metrics": quality_metrics}
                        if should_run_quality_filter(args) and quality_metrics is not None
                        else {}
                    ),
                },
            )
            stats["success"] += 1

        print(
            f"[build shard {args.shard_id}] parquet={Path(parquet_path).name} "
            f"requested={stats['requested']} success={stats['success']}"
        )

    write_json(
        summary_path,
        {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "selected_manifest": str(args.selected_manifest),
            "cache_root": str(cache_root),
            "stats": dict(stats),
        },
    )


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_metadata(metadata_path: Path, stems: Sequence[str]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write("sha256\n")
        for stem in stems:
            handle.write(f"{stem}\n")


def run_finalize_dataset(args: argparse.Namespace) -> None:
    if args.val_count >= args.target_count:
        raise ValueError("val-count must be smaller than target-count")

    cache_root = Path(args.cache_root)
    build_record_paths = sorted(cache_root.glob("build_records_shard*.jsonl"))
    if not build_record_paths:
        raise FileNotFoundError(f"No build record shards found under {cache_root}")

    build_records: List[dict] = []
    for path in build_record_paths:
        build_records.extend(load_jsonl(path))

    successes = [record for record in build_records if record["status"] in {"success", "cached"}]
    successes.sort(key=lambda item: item["rank"])
    if len(successes) < args.target_count:
        raise ValueError(f"Need {args.target_count} successful tensors, found {len(successes)}")

    val_records = successes[: args.val_count]
    train_records = successes[args.val_count : args.target_count]

    output_root = Path(args.output_root)
    train_dir = output_root / "train" / "data"
    val_dir = output_root / "val" / "data"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for record in val_records:
        src = Path(record["cache_path"])
        dst = val_dir / f"{record['stem']}.pt"
        hardlink_or_copy(src, dst)

    for record in train_records:
        src = Path(record["cache_path"])
        dst = train_dir / f"{record['stem']}.pt"
        hardlink_or_copy(src, dst)

    write_metadata(output_root / "val" / "metadata.csv", [record["stem"] for record in val_records])
    write_metadata(output_root / "train" / "metadata.csv", [record["stem"] for record in train_records])

    final_records_path = output_root / "final_records.jsonl"
    if final_records_path.exists():
        final_records_path.unlink()
    for split, records in (("val", val_records), ("train", train_records)):
        for record in records:
            final_record = dict(record)
            final_record["split"] = split
            write_jsonl(final_records_path, final_record)

    write_json(
        output_root / "manifest.json",
        {
            "source_manifest": str(args.selected_manifest),
            "cache_root": str(cache_root),
            "output_root": str(output_root),
            "target_count": args.target_count,
            "val_count": args.val_count,
            "train_count": args.target_count - args.val_count,
            "resolution": args.resolution,
            "n_sample_points": args.n_sample_points,
            "class_order": CLASS_ORDER,
            "official_parse": True,
            "official_prefilter": True,
            "coordinate_system": "NCS inputs restored to WCS with bbox before voxelization",
            "selection_policy": "First successful tensors in deterministic parquet,row order",
        },
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "scan":
        run_scan(args)
    elif args.command == "finalize-selection":
        run_finalize_selection(args)
    elif args.command == "build":
        run_build(args)
    elif args.command == "finalize-dataset":
        run_finalize_dataset(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()