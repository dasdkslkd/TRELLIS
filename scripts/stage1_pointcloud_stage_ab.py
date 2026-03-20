#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial import cKDTree


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
INDEX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_ORDER)}
ANALYTIC_FACE_MAP = {
    0: "PLANE",
    1: "CYLINDER",
    2: "CONE",
    3: "SPHERE",
    4: "TORUS",
}

DEFAULT_SELECTED_MANIFEST = Path(
    "/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/_pipeline/selected_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/stage1_pointcloud_stage_ab"
)


@dataclass(frozen=True)
class StageAPoints:
    points_xyz: np.ndarray
    point_voxel_id: np.ndarray
    point_class_id: np.ndarray
    point_voxel_coord: np.ndarray
    point_slot_id: np.ndarray


@dataclass(frozen=True)
class StageBPoints:
    points_xyz: np.ndarray
    point_class_id: np.ndarray
    point_voxel_coord: np.ndarray
    point_normals: np.ndarray
    point_curvature: np.ndarray
    primitive_radius: np.ndarray
    normal_valid_mask: np.ndarray
    source_count: np.ndarray


@dataclass(frozen=True)
class ValidationMatches:
    matched_gt_points: np.ndarray
    matched_gt_normals: np.ndarray
    nearest_distance: np.ndarray
    normal_angle_deg: np.ndarray
    evaluated_point_mask: np.ndarray


@dataclass(frozen=True)
class StageCPatches:
    point_patch_id: np.ndarray
    point_patch_class_id: np.ndarray
    patch_class_id: np.ndarray
    patch_sizes: np.ndarray
    primitive_radius: np.ndarray


@dataclass(frozen=True)
class PatchValidationMatches:
    matched_gt_face_id: np.ndarray
    evaluated_point_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Implement stage A/B for ABC1M stage1 tensors and validate PCA normals against raw parquet normals."
    )
    parser.add_argument(
        "--input-pts",
        type=Path,
        nargs="+",
        required=True,
        help="One or more stage1 sparse tensor .pt files.",
    )
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=DEFAULT_SELECTED_MANIFEST,
        help="JSONL manifest that maps stage1 stems to parquet path and row index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for stage A/B outputs and validation reports.",
    )
    parser.add_argument(
        "--stage1-sample-points",
        type=int,
        default=24,
        help="Number of sampled points stored per occupied voxel in the stage1 tensor.",
    )
    parser.add_argument(
        "--quantization",
        type=float,
        default=1e-5,
        help="Quantization step used for point deduplication.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=24,
        help="Neighborhood size used for PCA normal estimation.",
    )
    parser.add_argument(
        "--boundary-neighbors",
        type=int,
        default=24,
        help="Neighborhood size used when estimating boundary normals from all cleaned points.",
    )
    parser.add_argument(
        "--primitive-refine-neighbors",
        type=int,
        default=48,
        help="Candidate neighborhood size used when refining cylinder and torus normals with radius consistency.",
    )
    parser.add_argument(
        "--primitive-radius-angle-min-deg",
        type=float,
        default=3.0,
        help="Minimum normal angle used when deriving local primitive radius estimates for cylinder and torus points.",
    )
    parser.add_argument(
        "--primitive-radius-tolerance",
        type=float,
        default=0.35,
        help="Maximum relative radius difference allowed when refining cylinder and torus neighborhoods.",
    )
    parser.add_argument(
        "--enable-stage-c",
        action="store_true",
        help="Run minimal stage C patch clustering after stage B.",
    )
    parser.add_argument(
        "--patch-neighbors",
        type=int,
        default=16,
        help="KNN size used to build the stage C within-class adjacency graph.",
    )
    parser.add_argument(
        "--patch-radius-scale",
        type=float,
        default=2.0,
        help="Adaptive spatial radius multiplier applied to local KNN scale when building patch edges.",
    )
    parser.add_argument(
        "--patch-normal-angle-deg",
        type=float,
        default=15.0,
        help="Maximum normal angle for a valid patch edge.",
    )
    parser.add_argument(
        "--patch-curvature-diff",
        type=float,
        default=0.02,
        help="Maximum curvature difference allowed for BSPLINE patch edges.",
    )
    parser.add_argument(
        "--patch-voxel-gap",
        type=int,
        default=2,
        help="Maximum Chebyshev voxel distance allowed when linking two points into one patch.",
    )
    parser.add_argument(
        "--min-patch-size",
        type=int,
        default=32,
        help="Minimum connected-component size kept as a valid patch. Smaller components stay unlabeled.",
    )
    parser.add_argument(
        "--save-stage-a",
        action="store_true",
        help="Save stage A raw labeled points as an .npz file.",
    )
    parser.add_argument(
        "--save-stage-b",
        action="store_true",
        help="Save stage B cleaned points and normals as an .npz file.",
    )
    parser.add_argument(
        "--save-validation-matches",
        action="store_true",
        help="Save matched ground-truth points, normals, distances, and angles as an .npz file.",
    )
    parser.add_argument(
        "--save-stage-c",
        action="store_true",
        help="Save stage C patch ids and per-patch metadata as an .npz file.",
    )
    return parser.parse_args()


def decode_npy_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def normalize_face_mask(face_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(face_mask, dtype=bool)
    while mask.ndim > 3:
        mask = mask[..., 0]
    if mask.ndim == 4:
        mask = mask[..., 0]
    return mask


def map_face_type(code: int) -> str:
    return ANALYTIC_FACE_MAP.get(int(code), "BSPLINE_SURFACE")


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.maximum(norms, eps)
    return vectors / safe


def quantize_points(points: np.ndarray, step: float) -> np.ndarray:
    return np.round(np.asarray(points, dtype=np.float64) / step).astype(np.int64)


def normal_angle_deg(normals_a: np.ndarray, normals_b: np.ndarray) -> np.ndarray:
    dots = np.sum(normalize_vectors(normals_a) * normalize_vectors(normals_b), axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.degrees(np.arccos(dots)).astype(np.float32)


def load_sparse_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor)}")
    if not tensor.is_sparse:
        raise ValueError(f"Expected a sparse COO tensor in {path}, got a dense tensor")
    return tensor.coalesce()


def tensor_to_labeled_points(tensor: torch.Tensor, sample_points: int) -> StageAPoints:
    indices = tensor.indices().t().cpu().numpy().astype(np.int32, copy=False)
    values = tensor.values().cpu().numpy().astype(np.float32, copy=False)

    if values.shape[1] < 8 + sample_points * 3:
        raise ValueError(
            f"Tensor has {values.shape[1]} channels, but {8 + sample_points * 3} are required for {sample_points} samples"
        )

    class_id = np.argmax(values[:, 1:8], axis=1).astype(np.int32) if len(values) else np.zeros((0,), dtype=np.int32)
    sampled_xyz = values[:, 8:8 + sample_points * 3].reshape(-1, sample_points, 3)

    points_xyz: List[np.ndarray] = []
    point_voxel_id: List[np.ndarray] = []
    point_class_id: List[np.ndarray] = []
    point_voxel_coord: List[np.ndarray] = []
    point_slot_id: List[np.ndarray] = []

    for voxel_id, voxel_points in enumerate(sampled_xyz):
        valid_mask = np.isfinite(voxel_points).all(axis=1)
        valid_mask &= np.logical_and(voxel_points >= -1.05, voxel_points <= 1.05).all(axis=1)
        if not np.any(valid_mask):
            continue
        valid_points = voxel_points[valid_mask]
        slot_ids = np.nonzero(valid_mask)[0].astype(np.int32)

        local_quant = quantize_points(valid_points, 1e-6)
        _, unique_indices = np.unique(local_quant, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        valid_points = valid_points[unique_indices]
        slot_ids = slot_ids[unique_indices]
        if len(valid_points) == 0:
            continue

        points_xyz.append(valid_points.astype(np.float32, copy=False))
        point_voxel_id.append(np.full((len(valid_points),), voxel_id, dtype=np.int32))
        point_class_id.append(np.full((len(valid_points),), class_id[voxel_id], dtype=np.int32))
        point_voxel_coord.append(np.repeat(indices[voxel_id][None, :], len(valid_points), axis=0).astype(np.int32, copy=False))
        point_slot_id.append(slot_ids)

    if not points_xyz:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_i32 = np.zeros((0,), dtype=np.int32)
        return StageAPoints(
            points_xyz=empty_xyz,
            point_voxel_id=empty_i32,
            point_class_id=empty_i32,
            point_voxel_coord=np.zeros((0, 3), dtype=np.int32),
            point_slot_id=empty_i32,
        )

    return StageAPoints(
        points_xyz=np.concatenate(points_xyz, axis=0),
        point_voxel_id=np.concatenate(point_voxel_id, axis=0),
        point_class_id=np.concatenate(point_class_id, axis=0),
        point_voxel_coord=np.concatenate(point_voxel_coord, axis=0),
        point_slot_id=np.concatenate(point_slot_id, axis=0),
    )


def choose_group_class(class_values: np.ndarray) -> int:
    counts = np.bincount(class_values, minlength=len(CLASS_ORDER))
    non_boundary_counts = counts.copy()
    non_boundary_counts[CLASS_TO_INDEX["BOUNDARY"]] = 0
    if non_boundary_counts.sum() > 0:
        return int(np.argmax(non_boundary_counts))
    return int(np.argmax(counts))


def choose_group_voxel_coord(voxel_coords: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(np.asarray(voxel_coords, dtype=np.int32), axis=0, return_counts=True)
    return unique[int(np.argmax(counts))].astype(np.int32, copy=False)


def deduplicate_points_global(
    points_xyz: np.ndarray,
    point_class_id: np.ndarray,
    point_voxel_coord: np.ndarray,
    quantization: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points_xyz) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    quantized = quantize_points(points_xyz, quantization)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    quantized = quantized[order]
    sorted_points = points_xyz[order]
    sorted_class = point_class_id[order]
    sorted_voxel_coord = point_voxel_coord[order]

    unique_keys, starts = np.unique(quantized, axis=0, return_index=True)
    del unique_keys
    ends = np.concatenate([starts[1:], np.array([len(sorted_points)], dtype=np.int64)])

    clean_points = np.zeros((len(starts), 3), dtype=np.float32)
    clean_class = np.zeros((len(starts),), dtype=np.int32)
    clean_voxel_coord = np.zeros((len(starts), 3), dtype=np.int32)
    source_count = np.zeros((len(starts),), dtype=np.int32)

    for output_idx, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        group_points = sorted_points[start:end]
        group_class = sorted_class[start:end]
        group_voxel_coord = sorted_voxel_coord[start:end]
        clean_points[output_idx] = group_points.mean(axis=0, dtype=np.float64).astype(np.float32)
        clean_class[output_idx] = choose_group_class(group_class)
        clean_voxel_coord[output_idx] = choose_group_voxel_coord(group_voxel_coord)
        source_count[output_idx] = end - start

    return clean_points, clean_class, clean_voxel_coord, source_count


def compute_knn(points_xyz: np.ndarray, neighbors: int) -> Tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(points_xyz)
    effective_k = min(max(neighbors, 2), len(points_xyz))
    distances, indices = tree.query(points_xyz, k=effective_k)
    if effective_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    local_scale = distances[:, -1].astype(np.float32, copy=False)
    local_scale = np.maximum(local_scale, 1e-6)
    return tree, distances, indices, local_scale


def estimate_local_primitive_radius(
    points_xyz: np.ndarray,
    point_normals: np.ndarray,
    neighbors: int,
    angle_min_deg: float,
) -> np.ndarray:
    primitive_radius = np.full((len(points_xyz),), np.nan, dtype=np.float32)
    if len(points_xyz) < 4:
        return primitive_radius

    _, distances, indices, _ = compute_knn(points_xyz, neighbors=neighbors)
    for point_idx in range(len(points_xyz)):
        nbr_idx = indices[point_idx, 1:]
        if len(nbr_idx) == 0:
            continue
        chord = distances[point_idx, 1:].astype(np.float64, copy=False)
        center_normal = np.repeat(point_normals[point_idx][None, :], len(nbr_idx), axis=0)
        neighbor_normals = point_normals[nbr_idx]
        angles = normal_angle_deg(center_normal, neighbor_normals).astype(np.float64)
        valid = angles >= angle_min_deg
        if not np.any(valid):
            continue
        theta = np.deg2rad(angles[valid])
        denom = np.maximum(2.0 * np.sin(theta * 0.5), 1e-6)
        radius = chord[valid] / denom
        radius = radius[np.isfinite(radius)]
        if len(radius) == 0:
            continue
        primitive_radius[point_idx] = float(np.median(radius))
    return primitive_radius


def refine_normals_with_radius_constraint(
    points_xyz: np.ndarray,
    point_normals: np.ndarray,
    point_curvature: np.ndarray,
    primitive_radius: np.ndarray,
    neighbors: int,
    radius_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    refined_normals = point_normals.copy()
    refined_curvature = point_curvature.copy()
    refined_valid = np.ones((len(points_xyz),), dtype=bool)
    if len(points_xyz) < 4:
        return refined_normals, refined_curvature, refined_valid

    _, _, indices, _ = compute_knn(points_xyz, neighbors=neighbors)
    for point_idx in range(len(points_xyz)):
        nbr_idx = np.asarray(indices[point_idx], dtype=np.int64)
        if len(nbr_idx) < 3:
            continue
        if np.isfinite(primitive_radius[point_idx]):
            nbr_radius = primitive_radius[nbr_idx]
            finite_mask = np.isfinite(nbr_radius)
            if np.any(finite_mask):
                rel_diff = np.abs(nbr_radius - primitive_radius[point_idx]) / np.maximum(
                    np.maximum(nbr_radius, primitive_radius[point_idx]),
                    1e-6,
                )
                keep = np.logical_or(~finite_mask, rel_diff <= radius_tolerance)
                nbr_idx = nbr_idx[keep]
        if len(nbr_idx) < 3:
            continue
        neighborhood = points_xyz[nbr_idx]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / float(max(len(neighborhood) - 1, 1))
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        normal = evecs[:, 0]
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-12:
            refined_valid[point_idx] = False
            continue
        normal = (normal / norm).astype(np.float32)
        if np.dot(normal, refined_normals[point_idx]) < 0.0:
            normal = -normal
        refined_normals[point_idx] = normal
        denom = float(np.sum(evals))
        refined_curvature[point_idx] = float(evals[0] / denom) if denom > 1e-12 else 0.0
    return refined_normals, refined_curvature, refined_valid


def build_connected_components(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
    parent = np.arange(num_nodes, dtype=np.int32)
    rank = np.zeros((num_nodes,), dtype=np.int8)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    def union(node_a: int, node_b: int) -> None:
        root_a = find(node_a)
        root_b = find(node_b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for node_a, node_b in edges:
        union(int(node_a), int(node_b))

    groups: Dict[int, List[int]] = {}
    for node in range(num_nodes):
        groups.setdefault(find(node), []).append(node)
    return [np.asarray(nodes, dtype=np.int32) for nodes in groups.values()]


def estimate_normals_for_indices(
    points_xyz: np.ndarray,
    target_indices: np.ndarray,
    query_points: np.ndarray,
    neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.zeros((len(target_indices), 3), dtype=np.float32)
    curvature = np.full((len(target_indices),), np.nan, dtype=np.float32)
    valid = np.zeros((len(target_indices),), dtype=bool)
    if len(points_xyz) < 3 or len(target_indices) == 0:
        return normals, curvature, valid

    effective_k = min(max(neighbors, 3), len(points_xyz))
    tree = cKDTree(points_xyz)
    _, knn = tree.query(query_points, k=effective_k)
    if effective_k == 1:
        knn = knn[:, None]

    for local_idx, nbr_idx in enumerate(np.asarray(knn)):
        neighborhood = points_xyz[np.asarray(nbr_idx, dtype=np.int64)]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / float(max(len(neighborhood) - 1, 1))
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        normal = evecs[:, 0]
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-12:
            continue
        normals[local_idx] = (normal / norm).astype(np.float32)
        denom = float(np.sum(evals))
        curvature[local_idx] = float(evals[0] / denom) if denom > 1e-12 else 0.0
        valid[local_idx] = True

    return normals, curvature, valid


def clean_and_estimate_normals(
    stage_a: StageAPoints,
    quantization: float,
    neighbors: int,
    boundary_neighbors: int,
    primitive_refine_neighbors: int,
    primitive_radius_angle_min_deg: float,
    primitive_radius_tolerance: float,
) -> StageBPoints:
    clean_points, clean_class, clean_voxel_coord, source_count = deduplicate_points_global(
        stage_a.points_xyz,
        stage_a.point_class_id,
        stage_a.point_voxel_coord,
        quantization=quantization,
    )

    point_normals = np.zeros((len(clean_points), 3), dtype=np.float32)
    point_curvature = np.full((len(clean_points),), np.nan, dtype=np.float32)
    primitive_radius = np.full((len(clean_points),), np.nan, dtype=np.float32)
    normal_valid_mask = np.zeros((len(clean_points),), dtype=bool)

    boundary_id = CLASS_TO_INDEX["BOUNDARY"]
    primitive_class_ids = {CLASS_TO_INDEX["CYLINDER"], CLASS_TO_INDEX["TORUS"]}
    for class_id in range(len(CLASS_ORDER)):
        target_idx = np.nonzero(clean_class == class_id)[0]
        if len(target_idx) == 0:
            continue
        if class_id == boundary_id:
            neighbor_points = clean_points
            query_points = clean_points[target_idx]
            class_normals, class_curvature, class_valid = estimate_normals_for_indices(
                neighbor_points,
                target_idx,
                query_points,
                neighbors=boundary_neighbors,
            )
        else:
            query_points = clean_points[target_idx]
            class_normals, class_curvature, class_valid = estimate_normals_for_indices(
                query_points,
                target_idx,
                query_points,
                neighbors=neighbors,
            )

        point_normals[target_idx] = class_normals
        point_curvature[target_idx] = class_curvature
        normal_valid_mask[target_idx] = class_valid

        if class_id in primitive_class_ids and np.count_nonzero(class_valid) >= 4:
            valid_local_idx = np.nonzero(class_valid)[0]
            valid_point_idx = target_idx[valid_local_idx]
            class_points = clean_points[valid_point_idx]
            class_normals_valid = normalize_vectors(class_normals[valid_local_idx])
            class_radius = estimate_local_primitive_radius(
                class_points,
                class_normals_valid,
                neighbors=primitive_refine_neighbors,
                angle_min_deg=primitive_radius_angle_min_deg,
            )
            primitive_radius[valid_point_idx] = class_radius
            refined_normals, refined_curvature, refined_valid = refine_normals_with_radius_constraint(
                class_points,
                class_normals_valid,
                class_curvature[valid_local_idx],
                class_radius,
                neighbors=primitive_refine_neighbors,
                radius_tolerance=primitive_radius_tolerance,
            )
            point_normals[valid_point_idx] = refined_normals
            point_curvature[valid_point_idx] = refined_curvature
            normal_valid_mask[valid_point_idx] = refined_valid

    return StageBPoints(
        points_xyz=clean_points,
        point_class_id=clean_class,
        point_voxel_coord=clean_voxel_coord,
        point_normals=point_normals,
        point_curvature=point_curvature,
        primitive_radius=primitive_radius,
        normal_valid_mask=normal_valid_mask,
        source_count=source_count,
    )


def cluster_surface_patches(
    stage_b: StageBPoints,
    patch_neighbors: int,
    patch_radius_scale: float,
    patch_normal_angle_deg: float,
    patch_curvature_diff: float,
    patch_voxel_gap: int,
    primitive_radius_tolerance: float,
    min_patch_size: int,
) -> StageCPatches:
    point_patch_id = np.full((len(stage_b.points_xyz),), -1, dtype=np.int32)
    point_patch_class_id = np.full((len(stage_b.points_xyz),), -1, dtype=np.int32)
    patch_class_ids: List[int] = []
    patch_sizes: List[int] = []
    next_patch_id = 0

    boundary_id = CLASS_TO_INDEX["BOUNDARY"]
    bspline_id = CLASS_TO_INDEX["BSPLINE_SURFACE"]
    primitive_class_ids = {CLASS_TO_INDEX["CYLINDER"], CLASS_TO_INDEX["TORUS"]}

    for class_id in range(len(CLASS_ORDER)):
        if class_id == boundary_id:
            continue
        point_idx = np.nonzero((stage_b.point_class_id == class_id) & stage_b.normal_valid_mask)[0]
        if len(point_idx) == 0:
            continue

        class_points = stage_b.points_xyz[point_idx]
        class_normals = normalize_vectors(stage_b.point_normals[point_idx])
        class_curvature = stage_b.point_curvature[point_idx]
        class_voxel_coord = stage_b.point_voxel_coord[point_idx]
        class_radius = stage_b.primitive_radius[point_idx]
        _, distances, indices, local_scale = compute_knn(class_points, neighbors=patch_neighbors + 1)
        edges: List[Tuple[int, int]] = []

        for local_i in range(len(point_idx)):
            for offset in range(1, indices.shape[1]):
                local_j = int(indices[local_i, offset])
                if local_j <= local_i:
                    continue
                spatial_distance = float(distances[local_i, offset])
                spatial_thresh = float(max(local_scale[local_i], local_scale[local_j]) * patch_radius_scale)
                if spatial_distance > spatial_thresh:
                    continue
                if np.max(np.abs(class_voxel_coord[local_i] - class_voxel_coord[local_j])) > patch_voxel_gap:
                    continue
                angle = float(normal_angle_deg(class_normals[local_i : local_i + 1], class_normals[local_j : local_j + 1])[0])
                if angle > patch_normal_angle_deg:
                    continue
                if class_id == bspline_id:
                    if np.isfinite(class_curvature[local_i]) and np.isfinite(class_curvature[local_j]):
                        if abs(float(class_curvature[local_i] - class_curvature[local_j])) > patch_curvature_diff:
                            continue
                if class_id in primitive_class_ids:
                    radius_i = class_radius[local_i]
                    radius_j = class_radius[local_j]
                    if np.isfinite(radius_i) and np.isfinite(radius_j):
                        rel_diff = abs(float(radius_i - radius_j)) / max(float(max(radius_i, radius_j)), 1e-6)
                        if rel_diff > primitive_radius_tolerance:
                            continue
                edges.append((local_i, local_j))

        components = build_connected_components(len(point_idx), edges)
        for component in components:
            if len(component) < min_patch_size:
                continue
            global_idx = point_idx[component]
            point_patch_id[global_idx] = next_patch_id
            point_patch_class_id[global_idx] = class_id
            patch_class_ids.append(class_id)
            patch_sizes.append(int(len(component)))
            next_patch_id += 1

    return StageCPatches(
        point_patch_id=point_patch_id,
        point_patch_class_id=point_patch_class_id,
        patch_class_id=np.asarray(patch_class_ids, dtype=np.int32),
        patch_sizes=np.asarray(patch_sizes, dtype=np.int32),
        primitive_radius=stage_b.primitive_radius,
    )


def load_manifest_records(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["stem"])] = record
    return records


def load_parquet_row(record: dict) -> dict:
    columns = [
        "stem",
        "face_points_world",
        "face_normals",
        "face_mask",
        "face_types",
    ]
    table = pq.read_table(record["parquet_path"], columns=columns).slice(int(record["row_index"]), 1)
    return {
        "stem": table["stem"][0].as_py(),
        "face_points_world": decode_npy_blob(table["face_points_world"][0].as_py()),
        "face_normals": decode_npy_blob(table["face_normals"][0].as_py()),
        "face_mask": decode_npy_blob(table["face_mask"][0].as_py()),
        "face_types": decode_npy_blob(table["face_types"][0].as_py()),
    }


def build_augmented_face_points_with_normals(row: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face_points = np.asarray(row["face_points_world"], dtype=np.float32)
    face_normals = normalize_vectors(np.asarray(row["face_normals"], dtype=np.float32))
    face_mask = normalize_face_mask(row["face_mask"])
    face_types = np.asarray(row["face_types"], dtype=np.int32)

    points_list: List[np.ndarray] = []
    normals_list: List[np.ndarray] = []
    class_list: List[np.ndarray] = []
    face_id_list: List[np.ndarray] = []

    for face_idx in range(face_points.shape[0]):
        class_name = map_face_type(int(face_types[face_idx]))
        class_id = CLASS_TO_INDEX[class_name]
        pts = face_points[face_idx]
        nrm = face_normals[face_idx]
        mask = face_mask[face_idx]
        if not np.any(mask):
            continue

        valid_points = pts[mask]
        valid_normals = nrm[mask]
        if len(valid_points) > 0:
            points_list.append(valid_points)
            normals_list.append(valid_normals)
            class_list.append(np.full((len(valid_points),), class_id, dtype=np.int32))
            face_id_list.append(np.full((len(valid_points),), face_idx, dtype=np.int32))

        full_quads = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
        if np.any(full_quads):
            quad_points = (
                pts[:-1, :-1] + pts[1:, :-1] + pts[:-1, 1:] + pts[1:, 1:]
            ) * 0.25
            quad_normals = normalize_vectors(
                nrm[:-1, :-1] + nrm[1:, :-1] + nrm[:-1, 1:] + nrm[1:, 1:]
            )
            quad_points = quad_points[full_quads]
            quad_normals = quad_normals[full_quads]
            points_list.append(quad_points)
            normals_list.append(quad_normals)
            class_list.append(np.full((len(quad_points),), class_id, dtype=np.int32))
            face_id_list.append(np.full((len(quad_points),), face_idx, dtype=np.int32))

        vertical_pairs = mask[:-1, :] & mask[1:, :]
        if np.any(vertical_pairs):
            vertical_points = (pts[:-1, :] + pts[1:, :]) * 0.5
            vertical_normals = normalize_vectors(nrm[:-1, :] + nrm[1:, :])
            vertical_points = vertical_points[vertical_pairs]
            vertical_normals = vertical_normals[vertical_pairs]
            points_list.append(vertical_points)
            normals_list.append(vertical_normals)
            class_list.append(np.full((len(vertical_points),), class_id, dtype=np.int32))
            face_id_list.append(np.full((len(vertical_points),), face_idx, dtype=np.int32))

        horizontal_pairs = mask[:, :-1] & mask[:, 1:]
        if np.any(horizontal_pairs):
            horizontal_points = (pts[:, :-1] + pts[:, 1:]) * 0.5
            horizontal_normals = normalize_vectors(nrm[:, :-1] + nrm[:, 1:])
            horizontal_points = horizontal_points[horizontal_pairs]
            horizontal_normals = horizontal_normals[horizontal_pairs]
            points_list.append(horizontal_points)
            normals_list.append(horizontal_normals)
            class_list.append(np.full((len(horizontal_points),), class_id, dtype=np.int32))
            face_id_list.append(np.full((len(horizontal_points),), face_idx, dtype=np.int32))

    if not points_list:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    return (
        np.concatenate(points_list, axis=0),
        normalize_vectors(np.concatenate(normals_list, axis=0)).astype(np.float32, copy=False),
        np.concatenate(class_list, axis=0),
        np.concatenate(face_id_list, axis=0),
    )


def summarize_metric(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def validate_normals_with_parquet(stage_b: StageBPoints, row: dict) -> Tuple[dict, ValidationMatches]:
    gt_points, gt_normals, gt_class_id, _ = build_augmented_face_points_with_normals(row)
    boundary_id = CLASS_TO_INDEX["BOUNDARY"]

    matched_gt_points = np.full_like(stage_b.points_xyz, np.nan, dtype=np.float32)
    matched_gt_normals = np.full_like(stage_b.point_normals, np.nan, dtype=np.float32)
    nearest_distance = np.full((len(stage_b.points_xyz),), np.nan, dtype=np.float32)
    normal_angle_values = np.full((len(stage_b.points_xyz),), np.nan, dtype=np.float32)
    evaluated_point_mask = np.zeros((len(stage_b.points_xyz),), dtype=bool)

    per_class_metrics: Dict[str, dict] = {}
    overall_angles: List[np.ndarray] = []
    overall_distances: List[np.ndarray] = []

    for class_id, class_name in INDEX_TO_CLASS.items():
        if class_id == boundary_id:
            continue

        point_mask = (stage_b.point_class_id == class_id) & stage_b.normal_valid_mask
        gt_mask = gt_class_id == class_id
        point_idx = np.nonzero(point_mask)[0]
        if len(point_idx) == 0 or not np.any(gt_mask):
            per_class_metrics[class_name] = {
                "point_count": int(len(point_idx)),
                "gt_count": int(np.count_nonzero(gt_mask)),
                "distance": summarize_metric(np.zeros((0,), dtype=np.float32)),
                "angle_deg": summarize_metric(np.zeros((0,), dtype=np.float32)),
                "angle_le_5deg_ratio": None,
                "angle_le_10deg_ratio": None,
                "angle_le_20deg_ratio": None,
            }
            continue

        class_points = stage_b.points_xyz[point_idx]
        class_normals = normalize_vectors(stage_b.point_normals[point_idx])
        class_gt_points = gt_points[gt_mask]
        class_gt_normals = gt_normals[gt_mask]
        tree = cKDTree(class_gt_points)
        distances, match_idx = tree.query(class_points, k=1)
        matched_points = class_gt_points[match_idx]
        matched_normals = class_gt_normals[match_idx]
        angles = normal_angle_deg(class_normals, matched_normals)

        matched_gt_points[point_idx] = matched_points.astype(np.float32, copy=False)
        matched_gt_normals[point_idx] = matched_normals.astype(np.float32, copy=False)
        nearest_distance[point_idx] = distances.astype(np.float32, copy=False)
        normal_angle_values[point_idx] = angles
        evaluated_point_mask[point_idx] = True

        overall_angles.append(angles)
        overall_distances.append(distances.astype(np.float32, copy=False))
        per_class_metrics[class_name] = {
            "point_count": int(len(point_idx)),
            "gt_count": int(len(class_gt_points)),
            "distance": summarize_metric(distances.astype(np.float32, copy=False)),
            "angle_deg": summarize_metric(angles),
            "angle_le_5deg_ratio": float(np.mean(angles <= 5.0)),
            "angle_le_10deg_ratio": float(np.mean(angles <= 10.0)),
            "angle_le_20deg_ratio": float(np.mean(angles <= 20.0)),
        }

    overall_angle_values = np.concatenate(overall_angles, axis=0) if overall_angles else np.zeros((0,), dtype=np.float32)
    overall_distance_values = np.concatenate(overall_distances, axis=0) if overall_distances else np.zeros((0,), dtype=np.float32)

    summary = {
        "evaluated_point_count": int(np.count_nonzero(evaluated_point_mask)),
        "stage_b_point_count": int(len(stage_b.points_xyz)),
        "non_boundary_point_count": int(np.count_nonzero(stage_b.point_class_id != boundary_id)),
        "distance": summarize_metric(overall_distance_values),
        "angle_deg": summarize_metric(overall_angle_values),
        "angle_le_5deg_ratio": float(np.mean(overall_angle_values <= 5.0)) if len(overall_angle_values) else None,
        "angle_le_10deg_ratio": float(np.mean(overall_angle_values <= 10.0)) if len(overall_angle_values) else None,
        "angle_le_20deg_ratio": float(np.mean(overall_angle_values <= 20.0)) if len(overall_angle_values) else None,
        "per_class": per_class_metrics,
    }
    matches = ValidationMatches(
        matched_gt_points=matched_gt_points,
        matched_gt_normals=matched_gt_normals,
        nearest_distance=nearest_distance,
        normal_angle_deg=normal_angle_values,
        evaluated_point_mask=evaluated_point_mask,
    )
    return summary, matches


def validate_patches_with_parquet(stage_b: StageBPoints, stage_c: StageCPatches, row: dict) -> Tuple[dict, PatchValidationMatches]:
    gt_points, _, gt_class_id, gt_face_id = build_augmented_face_points_with_normals(row)
    matched_gt_face_id = np.full((len(stage_b.points_xyz),), -1, dtype=np.int32)
    evaluated_point_mask = np.zeros((len(stage_b.points_xyz),), dtype=bool)
    boundary_id = CLASS_TO_INDEX["BOUNDARY"]

    for class_id in range(len(CLASS_ORDER)):
        if class_id == boundary_id:
            continue
        point_idx = np.nonzero(
            (stage_b.point_class_id == class_id)
            & (stage_c.point_patch_id >= 0)
            & stage_b.normal_valid_mask
        )[0]
        gt_mask = gt_class_id == class_id
        if len(point_idx) == 0 or not np.any(gt_mask):
            continue
        class_points = stage_b.points_xyz[point_idx]
        class_gt_points = gt_points[gt_mask]
        class_gt_face_id = gt_face_id[gt_mask]
        tree = cKDTree(class_gt_points)
        _, match_idx = tree.query(class_points, k=1)
        matched_gt_face_id[point_idx] = class_gt_face_id[match_idx].astype(np.int32, copy=False)
        evaluated_point_mask[point_idx] = True

    patch_metrics: List[dict] = []
    valid_patch_ids = np.unique(stage_c.point_patch_id[stage_c.point_patch_id >= 0])
    purity_values: List[float] = []
    purity_weights: List[int] = []
    for patch_id in valid_patch_ids.tolist():
        point_mask = (stage_c.point_patch_id == patch_id) & evaluated_point_mask
        face_ids = matched_gt_face_id[point_mask]
        if len(face_ids) == 0:
            continue
        unique_faces, counts = np.unique(face_ids, return_counts=True)
        best_idx = int(np.argmax(counts))
        purity = float(counts[best_idx] / counts.sum())
        patch_metrics.append(
            {
                "patch_id": int(patch_id),
                "class_name": INDEX_TO_CLASS[int(stage_c.patch_class_id[patch_id])],
                "size": int(np.count_nonzero(stage_c.point_patch_id == patch_id)),
                "evaluated_size": int(counts.sum()),
                "majority_face_id": int(unique_faces[best_idx]),
                "purity": purity,
            }
        )
        purity_values.append(purity)
        purity_weights.append(int(counts.sum()))

    face_metrics: List[dict] = []
    gt_non_boundary_faces = np.unique(gt_face_id[gt_class_id != boundary_id])
    completeness_values: List[float] = []
    completeness_weights: List[int] = []
    for face_id in gt_non_boundary_faces.tolist():
        point_mask = evaluated_point_mask & (matched_gt_face_id == face_id)
        if not np.any(point_mask):
            continue
        patch_ids, counts = np.unique(stage_c.point_patch_id[point_mask], return_counts=True)
        valid_mask = patch_ids >= 0
        patch_ids = patch_ids[valid_mask]
        counts = counts[valid_mask]
        if len(patch_ids) == 0:
            continue
        best_idx = int(np.argmax(counts))
        gt_point_count = int(np.count_nonzero(gt_face_id == face_id))
        completeness = float(counts[best_idx] / max(gt_point_count, 1))
        face_metrics.append(
            {
                "face_id": int(face_id),
                "best_patch_id": int(patch_ids[best_idx]),
                "matched_points": int(counts[best_idx]),
                "gt_point_count": gt_point_count,
                "completeness": completeness,
            }
        )
        completeness_values.append(completeness)
        completeness_weights.append(gt_point_count)

    summary = {
        "patch_count": int(len(stage_c.patch_class_id)),
        "evaluated_patch_count": int(len(patch_metrics)),
        "gt_face_count": int(len(gt_non_boundary_faces)),
        "mean_patch_purity": float(np.mean(purity_values)) if purity_values else None,
        "weighted_patch_purity": (
            float(np.average(np.asarray(purity_values, dtype=np.float32), weights=np.asarray(purity_weights, dtype=np.float32)))
            if purity_values
            else None
        ),
        "mean_face_completeness": float(np.mean(completeness_values)) if completeness_values else None,
        "weighted_face_completeness": (
            float(np.average(np.asarray(completeness_values, dtype=np.float32), weights=np.asarray(completeness_weights, dtype=np.float32)))
            if completeness_values
            else None
        ),
        "unlabeled_point_count": int(np.count_nonzero(stage_c.point_patch_id < 0)),
        "patch_metrics": patch_metrics,
        "face_metrics": face_metrics,
    }
    return summary, PatchValidationMatches(matched_gt_face_id=matched_gt_face_id, evaluated_point_mask=evaluated_point_mask)


def save_stage_a(path: Path, stage_a: StageAPoints) -> None:
    np.savez_compressed(
        path,
        points_xyz=stage_a.points_xyz,
        point_voxel_id=stage_a.point_voxel_id,
        point_class_id=stage_a.point_class_id,
        point_voxel_coord=stage_a.point_voxel_coord,
        point_slot_id=stage_a.point_slot_id,
    )


def save_stage_b(path: Path, stage_b: StageBPoints) -> None:
    np.savez_compressed(
        path,
        points_xyz=stage_b.points_xyz,
        point_class_id=stage_b.point_class_id,
        point_voxel_coord=stage_b.point_voxel_coord,
        point_normals=stage_b.point_normals,
        point_curvature=stage_b.point_curvature,
        primitive_radius=stage_b.primitive_radius,
        normal_valid_mask=stage_b.normal_valid_mask,
        source_count=stage_b.source_count,
    )


def save_validation_matches(path: Path, matches: ValidationMatches) -> None:
    np.savez_compressed(
        path,
        matched_gt_points=matches.matched_gt_points,
        matched_gt_normals=matches.matched_gt_normals,
        nearest_distance=matches.nearest_distance,
        normal_angle_deg=matches.normal_angle_deg,
        evaluated_point_mask=matches.evaluated_point_mask,
    )


def save_stage_c(path: Path, stage_c: StageCPatches) -> None:
    np.savez_compressed(
        path,
        point_patch_id=stage_c.point_patch_id,
        point_patch_class_id=stage_c.point_patch_class_id,
        patch_class_id=stage_c.patch_class_id,
        patch_sizes=stage_c.patch_sizes,
        primitive_radius=stage_c.primitive_radius,
    )


def build_stage_summary(
    stem: str,
    pt_path: Path,
    record: dict,
    stage_a: StageAPoints,
    stage_b: StageBPoints,
    validation_summary: dict,
    stage_c: Optional[StageCPatches],
    patch_validation_summary: Optional[dict],
    args: argparse.Namespace,
) -> dict:
    class_counts_stage_a = {
        name: int(np.count_nonzero(stage_a.point_class_id == idx))
        for idx, name in INDEX_TO_CLASS.items()
    }
    class_counts_stage_b = {
        name: int(np.count_nonzero(stage_b.point_class_id == idx))
        for idx, name in INDEX_TO_CLASS.items()
    }
    valid_normal_counts = {
        name: int(np.count_nonzero((stage_b.point_class_id == idx) & stage_b.normal_valid_mask))
        for idx, name in INDEX_TO_CLASS.items()
    }
    summary = {
        "stem": stem,
        "pt_path": str(pt_path),
        "parquet_path": str(record["parquet_path"]),
        "row_index": int(record["row_index"]),
        "stage_a": {
            "point_count": int(len(stage_a.points_xyz)),
            "class_counts": class_counts_stage_a,
        },
        "stage_b": {
            "point_count": int(len(stage_b.points_xyz)),
            "class_counts": class_counts_stage_b,
            "normal_valid_count": int(np.count_nonzero(stage_b.normal_valid_mask)),
            "valid_normal_counts": valid_normal_counts,
            "mean_source_count": float(np.mean(stage_b.source_count)) if len(stage_b.source_count) else None,
            "mean_curvature": float(np.nanmean(stage_b.point_curvature)) if np.isfinite(stage_b.point_curvature).any() else None,
            "mean_primitive_radius": float(np.nanmean(stage_b.primitive_radius)) if np.isfinite(stage_b.primitive_radius).any() else None,
        },
        "validation": validation_summary,
        "config": {
            "stage1_sample_points": int(args.stage1_sample_points),
            "quantization": float(args.quantization),
            "neighbors": int(args.neighbors),
            "boundary_neighbors": int(args.boundary_neighbors),
            "primitive_refine_neighbors": int(args.primitive_refine_neighbors),
            "primitive_radius_angle_min_deg": float(args.primitive_radius_angle_min_deg),
            "primitive_radius_tolerance": float(args.primitive_radius_tolerance),
            "enable_stage_c": bool(args.enable_stage_c),
            "patch_neighbors": int(args.patch_neighbors),
            "patch_radius_scale": float(args.patch_radius_scale),
            "patch_normal_angle_deg": float(args.patch_normal_angle_deg),
            "patch_curvature_diff": float(args.patch_curvature_diff),
            "patch_voxel_gap": int(args.patch_voxel_gap),
            "min_patch_size": int(args.min_patch_size),
        },
    }

    if stage_c is not None:
        patch_class_counts = {
            name: int(np.count_nonzero(stage_c.point_patch_class_id == idx))
            for idx, name in INDEX_TO_CLASS.items()
            if idx != CLASS_TO_INDEX["BOUNDARY"]
        }
        summary["stage_c"] = {
            "patch_count": int(len(stage_c.patch_class_id)),
            "labeled_point_count": int(np.count_nonzero(stage_c.point_patch_id >= 0)),
            "unlabeled_point_count": int(np.count_nonzero(stage_c.point_patch_id < 0)),
            "patch_class_counts": patch_class_counts,
            "mean_patch_size": float(np.mean(stage_c.patch_sizes)) if len(stage_c.patch_sizes) else None,
            "median_patch_size": float(np.median(stage_c.patch_sizes)) if len(stage_c.patch_sizes) else None,
        }

    if patch_validation_summary is not None:
        summary["patch_validation"] = patch_validation_summary

    return summary


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = load_manifest_records(args.selected_manifest)
    aggregate_summary: List[dict] = []

    for pt_path in args.input_pts:
        stem = pt_path.stem
        if stem not in manifest_records:
            raise KeyError(f"Stem {stem} was not found in {args.selected_manifest}")

        record = manifest_records[stem]
        tensor = load_sparse_tensor(pt_path)
        stage_a = tensor_to_labeled_points(tensor, sample_points=args.stage1_sample_points)
        stage_b = clean_and_estimate_normals(
            stage_a,
            quantization=args.quantization,
            neighbors=args.neighbors,
            boundary_neighbors=args.boundary_neighbors,
            primitive_refine_neighbors=args.primitive_refine_neighbors,
            primitive_radius_angle_min_deg=args.primitive_radius_angle_min_deg,
            primitive_radius_tolerance=args.primitive_radius_tolerance,
        )
        row = load_parquet_row(record)
        validation_summary, validation_matches = validate_normals_with_parquet(stage_b, row)
        stage_c = None
        patch_validation_summary = None
        if args.enable_stage_c:
            stage_c = cluster_surface_patches(
                stage_b,
                patch_neighbors=args.patch_neighbors,
                patch_radius_scale=args.patch_radius_scale,
                patch_normal_angle_deg=args.patch_normal_angle_deg,
                patch_curvature_diff=args.patch_curvature_diff,
                patch_voxel_gap=args.patch_voxel_gap,
                primitive_radius_tolerance=args.primitive_radius_tolerance,
                min_patch_size=args.min_patch_size,
            )
            patch_validation_summary, _ = validate_patches_with_parquet(stage_b, stage_c, row)

        sample_dir = args.output_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        if args.save_stage_a:
            stage_a_path = sample_dir / "stage_a_points.npz"
            ensure_parent(stage_a_path)
            save_stage_a(stage_a_path, stage_a)

        if args.save_stage_b:
            stage_b_path = sample_dir / "stage_b_points.npz"
            ensure_parent(stage_b_path)
            save_stage_b(stage_b_path, stage_b)

        if args.save_validation_matches:
            matches_path = sample_dir / "normal_validation_matches.npz"
            ensure_parent(matches_path)
            save_validation_matches(matches_path, validation_matches)

        if args.save_stage_c and stage_c is not None:
            stage_c_path = sample_dir / "stage_c_patches.npz"
            ensure_parent(stage_c_path)
            save_stage_c(stage_c_path, stage_c)

        summary = build_stage_summary(
            stem,
            pt_path,
            record,
            stage_a,
            stage_b,
            validation_summary,
            stage_c,
            patch_validation_summary,
            args,
        )
        summary_path = sample_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        aggregate_summary.append(summary)
        print(json.dumps({
            "stem": stem,
            "stage_a_points": summary["stage_a"]["point_count"],
            "stage_b_points": summary["stage_b"]["point_count"],
            "evaluated_point_count": summary["validation"]["evaluated_point_count"],
            "mean_angle_deg": summary["validation"]["angle_deg"]["mean"],
            "median_angle_deg": summary["validation"]["angle_deg"]["median"],
            "p90_angle_deg": summary["validation"]["angle_deg"]["p90"],
            "mean_nn_distance": summary["validation"]["distance"]["mean"],
            "patch_count": None if stage_c is None else summary["stage_c"]["patch_count"],
            "weighted_patch_purity": None if patch_validation_summary is None else patch_validation_summary["weighted_patch_purity"],
            "weighted_face_completeness": None if patch_validation_summary is None else patch_validation_summary["weighted_face_completeness"],
        }, ensure_ascii=False))

    aggregate_path = args.output_dir / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()