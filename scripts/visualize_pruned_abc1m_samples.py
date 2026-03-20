#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_abc1m_raw_vs_pointcloud import (  # noqa: E402
    SampleData,
    load_selected_samples,
    render_overview,
    render_single_sample,
)


DEFAULT_PRUNED_JSONL = Path(
    "/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2/_pipeline/pruned_filtered_samples.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/public/home/pb22000140/TRELLIS-new/TRELLIS-main/outputs/visualizations/abc1m_pruned_samples_world"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample filtered ABC-1M records by reason and visualize them in world coordinates."
    )
    parser.add_argument("--pruned-jsonl", type=Path, default=DEFAULT_PRUNED_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-reason", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260319)
    parser.add_argument("--reasons", nargs="*", default=[])
    parser.add_argument("--bucket-reason", type=str, default="")
    parser.add_argument(
        "--largest-component-ratio-bins",
        type=float,
        nargs="*",
        default=[],
        help="Bucket starts for largest_component_ratio, e.g. 0.5 0.6 0.7 0.8 0.9",
    )
    parser.add_argument(
        "--largest-component-ratio-max",
        type=float,
        default=0.95,
        help="Upper bound for the last largest_component_ratio bucket.",
    )
    parser.add_argument("--max-pointcloud-points", type=int, default=16000)
    parser.add_argument("--point-size", type=float, default=0.45)
    parser.add_argument("--mesh-alpha", type=float, default=0.95)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def choose_samples(records: Sequence[dict], samples_per_reason: int, seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["reason"])].append(record)

    chosen: Dict[str, List[dict]] = {}
    for reason, items in sorted(grouped.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        chosen[reason] = sorted(
            shuffled[: min(samples_per_reason, len(shuffled))],
            key=lambda item: (item["parquet_path"], int(item["row_index"]), item["stem"]),
        )
    return chosen


def format_ratio_label(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_ratio_bucket_ranges(
    starts: Sequence[float],
    upper_bound: float,
) -> List[Tuple[str, float, float]]:
    if not starts:
        raise ValueError("largest-component-ratio-bins must not be empty in bucket mode")
    ordered = sorted(float(value) for value in starts)
    ranges: List[Tuple[str, float, float]] = []
    for idx, start in enumerate(ordered):
        end = ordered[idx + 1] if idx + 1 < len(ordered) else upper_bound
        if end <= start:
            raise ValueError("largest_component_ratio bucket boundaries must be strictly increasing")
        label = f"largest_component_ratio_{format_ratio_label(start)}_to_{format_ratio_label(end)}"
        ranges.append((label, start, end))
    return ranges


def choose_samples_by_ratio_buckets(
    records: Sequence[dict],
    bucket_reason: str,
    bucket_starts: Sequence[float],
    upper_bound: float,
    samples_per_bucket: int,
    seed: int,
) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    bucket_ranges = build_ratio_bucket_ranges(bucket_starts, upper_bound)
    chosen: Dict[str, List[dict]] = {}
    filtered = [record for record in records if record.get("reason") == bucket_reason]
    for label, start, end in bucket_ranges:
        bucket_records = []
        for record in filtered:
            ratio = float(record.get("quality_metrics", {}).get("largest_component_ratio", math.nan))
            if not math.isfinite(ratio):
                continue
            is_last = math.isclose(end, upper_bound)
            if start <= ratio < end or (is_last and start <= ratio <= end):
                bucket_records.append(record)
        shuffled = list(bucket_records)
        rng.shuffle(shuffled)
        chosen[label] = sorted(
            shuffled[: min(samples_per_bucket, len(shuffled))],
            key=lambda item: (item["parquet_path"], int(item["row_index"]), item["stem"]),
        )
    return chosen


def group_by_parquet(records: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["parquet_path"])].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: int(item["row_index"]))
    return grouped


def load_samples(records: Sequence[dict]) -> List[SampleData]:
    grouped = group_by_parquet(records)
    samples: List[SampleData] = []
    for parquet_path, items in sorted(grouped.items()):
        row_indices = [int(item["row_index"]) for item in items]
        loaded = load_selected_samples(Path(parquet_path), "world", row_indices, stems=[])
        samples.extend(loaded)
    sample_by_stem = {sample.stem: sample for sample in samples}
    return [sample_by_stem[item["stem"]] for item in records if item["stem"] in sample_by_stem]


def write_reason_manifest(path: Path, reason: str, records: Sequence[dict], metadata: Optional[dict] = None) -> None:
    payload = {
        "reason": reason,
        "sample_count": len(records),
        "samples": records,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.pruned_jsonl)
    if args.reasons:
        wanted = set(args.reasons)
        records = [record for record in records if record["reason"] in wanted]
    if not records:
        raise ValueError("No filtered records matched the requested reasons.")

    bucket_mode = bool(args.bucket_reason)
    if bucket_mode:
        selected_by_reason = choose_samples_by_ratio_buckets(
            records,
            bucket_reason=args.bucket_reason,
            bucket_starts=args.largest_component_ratio_bins,
            upper_bound=args.largest_component_ratio_max,
            samples_per_bucket=args.samples_per_reason,
            seed=args.seed,
        )
    else:
        selected_by_reason = choose_samples(records, args.samples_per_reason, args.seed)

    summary = {
        "pruned_jsonl": str(args.pruned_jsonl),
        "output_dir": str(args.output_dir),
        "samples_per_reason": args.samples_per_reason,
        "seed": args.seed,
        "reasons": {reason: len(items) for reason, items in selected_by_reason.items()},
        "coord_space": "world",
        "bucket_reason": args.bucket_reason,
        "largest_component_ratio_bins": list(args.largest_component_ratio_bins),
        "largest_component_ratio_max": args.largest_component_ratio_max,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for reason, selected_records in selected_by_reason.items():
        reason_dir = args.output_dir / reason
        reason_dir.mkdir(parents=True, exist_ok=True)
        if not selected_records:
            write_reason_manifest(
                reason_dir / "samples.json",
                reason,
                [],
                metadata={"bucket_reason": args.bucket_reason, "empty": True},
            )
            print(f"[{reason}] no samples matched")
            continue
        samples = load_samples(selected_records)
        overview_path = reason_dir / "overview.png"
        render_overview(samples, overview_path, args.max_pointcloud_points, args.point_size, args.mesh_alpha)
        for sample in samples:
            render_single_sample(
                sample,
                reason_dir / f"{sample.stem}.png",
                args.max_pointcloud_points,
                args.point_size,
                args.mesh_alpha,
            )
        write_reason_manifest(
            reason_dir / "samples.json",
            reason,
            selected_records,
            metadata={
                "bucket_reason": args.bucket_reason,
                "bucket_mode": bucket_mode,
            },
        )
        print(f"[{reason}] wrote {len(samples)} samples to {reason_dir}")


if __name__ == "__main__":
    main()