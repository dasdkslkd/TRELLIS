#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_PARQUET = Path('/public/home/pb22000140/xjn/datasets/ABC-1M/test/0000.parquet')
DEFAULT_OUTPUT_DIR = Path('/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/visualizations/abc1m_raw_vs_pointcloud')


@dataclass(frozen=True)
class SampleData:
    stem: str
    face_points: np.ndarray
    face_mask: np.ndarray
    edge_points: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize raw ABC-1M CAD surface grids against their sampled point clouds.'
    )
    parser.add_argument('--parquet', type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        '--row-indices',
        type=int,
        nargs='*',
        default=[0, 1, 2, 3],
        help='Row indices inside the parquet shard to visualize.',
    )
    parser.add_argument(
        '--stems',
        nargs='*',
        default=[],
        help='Optional stems to visualize. When provided, --row-indices is ignored.',
    )
    parser.add_argument(
        '--coord-space',
        choices=['normalized', 'world'],
        default='normalized',
        help='Coordinate space used for both surface mesh and point cloud.',
    )
    parser.add_argument('--max-pointcloud-points', type=int, default=16000)
    parser.add_argument('--point-size', type=float, default=0.45)
    parser.add_argument('--mesh-alpha', type=float, default=0.95)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def decode_npy_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def normalize_face_mask(face_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(face_mask, dtype=bool)
    while mask.ndim > 3:
        mask = mask[..., 0]
    return mask


def load_selected_samples(parquet_path: Path, coord_space: str, row_indices: Sequence[int], stems: Sequence[str]) -> List[SampleData]:
    coord_column = f'face_points_{coord_space}'
    edge_column = f'edge_points_{coord_space}'
    columns = ['stem', coord_column, 'face_mask', edge_column]
    table = pq.read_table(parquet_path, columns=columns)

    if stems:
        requested = set(stems)
        selected_rows = [idx for idx, stem in enumerate(table['stem'].to_pylist()) if stem in requested]
        missing = requested - {table['stem'][idx].as_py() for idx in selected_rows}
        if missing:
            raise KeyError(f'Stems not found in {parquet_path}: {sorted(missing)}')
    else:
        selected_rows = list(dict.fromkeys(int(idx) for idx in row_indices))

    samples: List[SampleData] = []
    for row_idx in selected_rows:
        if row_idx < 0 or row_idx >= table.num_rows:
            raise IndexError(f'Row index {row_idx} is out of range for {parquet_path} with {table.num_rows} rows')
        samples.append(
            SampleData(
                stem=table['stem'][row_idx].as_py(),
                face_points=decode_npy_blob(table[coord_column][row_idx].as_py()),
                face_mask=decode_npy_blob(table['face_mask'][row_idx].as_py()),
                edge_points=decode_npy_blob(table[edge_column][row_idx].as_py()),
            )
        )
    return samples


def triangulate_face_grid(face_points: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    triangles: List[np.ndarray] = []
    mask = normalize_face_mask(face_mask)

    for face_idx in range(face_points.shape[0]):
        pts = np.asarray(face_points[face_idx], dtype=np.float32)
        valid = mask[face_idx]
        while valid.ndim > 2:
            valid = valid[..., 0]
        rows, cols = valid.shape
        for row in range(rows - 1):
            for col in range(cols - 1):
                corners = np.asarray(
                    [pts[row, col], pts[row + 1, col], pts[row + 1, col + 1], pts[row, col + 1]],
                    dtype=np.float32,
                )
                corner_mask = np.asarray(
                    [valid[row, col], valid[row + 1, col], valid[row + 1, col + 1], valid[row, col + 1]],
                    dtype=bool,
                )
                active = corners[corner_mask]
                if active.shape[0] < 3:
                    continue
                if active.shape[0] == 3:
                    triangles.append(active)
                    continue
                triangles.append(corners[[0, 1, 2]])
                triangles.append(corners[[0, 2, 3]])
    if not triangles:
        return np.zeros((0, 3, 3), dtype=np.float32)
    return np.stack(triangles, axis=0)


def build_pointcloud(sample: SampleData) -> tuple[np.ndarray, np.ndarray]:
    mask = normalize_face_mask(sample.face_mask)
    face_points = np.asarray(sample.face_points, dtype=np.float32)[mask]
    edge_points = np.asarray(sample.edge_points, dtype=np.float32).reshape(-1, 3)

    if edge_points.size == 0:
        all_points = face_points
        point_labels = np.zeros((face_points.shape[0],), dtype=np.int32)
    else:
        all_points = np.concatenate([face_points, edge_points], axis=0)
        point_labels = np.concatenate(
            [
                np.zeros((face_points.shape[0],), dtype=np.int32),
                np.ones((edge_points.shape[0],), dtype=np.int32),
            ],
            axis=0,
        )
    return all_points, point_labels


def subsample_points(points: np.ndarray, labels: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= limit:
        return points, labels
    choose = np.linspace(0, points.shape[0] - 1, num=limit, dtype=np.int64)
    return points[choose], labels[choose]


def compute_bounds(sample: SampleData) -> tuple[np.ndarray, np.ndarray]:
    points, _ = build_pointcloud(sample)
    if points.size == 0:
        zeros = np.zeros((3,), dtype=np.float32)
        return zeros - 1.0, zeros + 1.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1e-3)
    return center - radius, center + radius


def style_axis(ax, title: str, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    lower, upper = bounds
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=312)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_facecolor('#f8fafc')


def render_mesh_panel(ax, sample: SampleData, mesh_alpha: float) -> None:
    triangles = triangulate_face_grid(sample.face_points, sample.face_mask)
    bounds = compute_bounds(sample)
    if triangles.shape[0] == 0:
        style_axis(ax, 'Raw CAD Surface (empty)', bounds)
        return
    mesh = Poly3DCollection(triangles, linewidths=0.02, edgecolors='none', alpha=mesh_alpha)
    mesh.set_facecolor('#7c9a92')
    ax.add_collection3d(mesh)
    style_axis(ax, 'Raw CAD Surface Grids', bounds)


def render_pointcloud_panel(ax, sample: SampleData, max_points: int, point_size: float) -> None:
    points, labels = build_pointcloud(sample)
    points, labels = subsample_points(points, labels, max_points)
    bounds = compute_bounds(sample)
    if points.shape[0] == 0:
        style_axis(ax, 'Sampled Point Cloud (empty)', bounds)
        return

    colors = np.where(labels[:, None] == 0, np.array([[66, 133, 244]]) / 255.0, np.array([[217, 119, 6]]) / 255.0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=point_size, alpha=0.9, depthshade=False)
    style_axis(ax, 'Sampled Point Cloud', bounds)


def render_overview(samples: Sequence[SampleData], output_path: Path, max_points: int, point_size: float, mesh_alpha: float) -> None:
    fig = plt.figure(figsize=(12, 5 * len(samples)), constrained_layout=True)
    for row_idx, sample in enumerate(samples):
        ax_mesh = fig.add_subplot(len(samples), 2, row_idx * 2 + 1, projection='3d')
        ax_cloud = fig.add_subplot(len(samples), 2, row_idx * 2 + 2, projection='3d')
        render_mesh_panel(ax_mesh, sample, mesh_alpha)
        render_pointcloud_panel(ax_cloud, sample, max_points, point_size)
        ax_mesh.text2D(0.02, 0.98, sample.stem, transform=ax_mesh.transAxes, fontsize=11, va='top')
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def render_single_sample(sample: SampleData, output_path: Path, max_points: int, point_size: float, mesh_alpha: float) -> None:
    fig = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    ax_mesh = fig.add_subplot(1, 2, 1, projection='3d')
    ax_cloud = fig.add_subplot(1, 2, 2, projection='3d')
    render_mesh_panel(ax_mesh, sample, mesh_alpha)
    render_pointcloud_panel(ax_cloud, sample, max_points, point_size)
    fig.suptitle(sample.stem, fontsize=14)
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def write_manifest(samples: Iterable[SampleData], output_path: Path, parquet_path: Path, coord_space: str) -> None:
    lines = [
        f'parquet: {parquet_path}',
        f'coord_space: {coord_space}',
        'legend: blue=face samples, orange=edge samples',
        'note: left panel triangulates raw ABC-1M face grids; this avoids unstable OCC BRep import in the current environment.',
        '',
        'samples:',
    ]
    for sample in samples:
        points, labels = build_pointcloud(sample)
        edge_count = int((labels == 1).sum())
        face_count = int((labels == 0).sum())
        lines.append(f'- {sample.stem}: face_points={face_count}, edge_points={edge_count}')
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_selected_samples(args.parquet, args.coord_space, args.row_indices, args.stems)

    overview_path = args.output_dir / 'overview.png'
    render_overview(samples, overview_path, args.max_pointcloud_points, args.point_size, args.mesh_alpha)

    for sample in samples:
        render_single_sample(
            sample,
            args.output_dir / f'{sample.stem}.png',
            args.max_pointcloud_points,
            args.point_size,
            args.mesh_alpha,
        )

    write_manifest(samples, args.output_dir / 'manifest.txt', args.parquet, args.coord_space)
    print(f'Wrote overview to {overview_path}')
    for sample in samples:
        print(f'Wrote sample figure: {args.output_dir / (sample.stem + ".png")}')


if __name__ == '__main__':
    main()