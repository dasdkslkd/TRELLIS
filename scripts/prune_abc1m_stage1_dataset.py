#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_toolkits.build_abc1m_sparse_structure_v2 import (  # noqa: E402
    build_prefilter_reason,
    build_quality_reason,
    iter_requested_rows,
    load_jsonl,
    validate_voxel_inputs,
    write_json,
    write_jsonl,
    write_metadata,
)


DEFAULT_DATASET_ROOT = Path(
    "/public/home/pb22000140/datasets/ABC1M_voxelized_brep_64_80ch_stage1_split_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune filtered samples in-place from an existing ABC1M stage1 dataset."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bit", type=int, default=10)
    parser.add_argument("--max-edge", type=int, default=1000)
    parser.add_argument("--quality-resolution", type=int, default=64)
    parser.add_argument("--max-voxel-components", type=int, default=32)
    parser.add_argument("--max-small-component-ratio", type=float, default=0.05)
    parser.add_argument("--low-ratio-min", type=float, default=0.9)
    parser.add_argument("--low-ratio-max", type=float, default=0.95)
    parser.add_argument("--low-ratio-component-count-threshold", type=int, default=16)
    parser.add_argument("--delete-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-stems", nargs="*", default=[])
    parser.add_argument("--progress-interval", type=int, default=2000)
    parser.add_argument("--delete-progress-interval", type=int, default=2000)
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def print_progress(prefix: str, completed: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    progress_pct = 100.0 * completed / total if total > 0 else 100.0
    eta_text = format_seconds(eta_seconds) if rate > 0 else "unknown"
    print(
        f"[{prefix}] {completed}/{total} ({progress_pct:.2f}%) "
        f"elapsed={format_seconds(elapsed)} rate={rate:.2f}/s ETA={eta_text}",
        flush=True,
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_final_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def backup_files(backup_dir: Path, paths: Sequence[Path]) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)


def split_order(split: str) -> int:
    if split == "val":
        return 0
    if split == "train":
        return 1
    return 2


def filter_current_records(
    final_records: Sequence[dict],
    only_stems: Sequence[str],
) -> List[dict]:
    if not only_stems:
        return list(final_records)
    allowed = set(only_stems)
    return [record for record in final_records if record["stem"] in allowed]


def evaluate_records(
    records_to_check: Sequence[dict],
    selected_by_rank: Dict[int, dict],
    args: argparse.Namespace,
) -> Tuple[List[dict], Counter]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records_to_check:
        selected = selected_by_rank.get(int(record["rank"]))
        if selected is None:
            raise KeyError(f"Missing selected manifest entry for rank={record['rank']}")
        merged = {
            "rank": int(record["rank"]),
            "stem": record["stem"],
            "split": record["split"],
            "cache_path": record.get("cache_path"),
            "parquet_path": selected["parquet_path"],
            "row_index": int(selected["row_index"]),
        }
        grouped[merged["parquet_path"]].append(merged)

    for items in grouped.values():
        items.sort(key=lambda item: item["row_index"])

    removed: List[dict] = []
    stats: Counter = Counter()
    total_records = len(records_to_check)
    start_time = time.time()
    print(
        f"[evaluate] start total_records={total_records} total_parquets={len(grouped)} "
        f"filter=max_voxel_components>{args.max_voxel_components}, "
        f"max_small_component_ratio>{args.max_small_component_ratio}, "
        f"low_ratio_range=[{args.low_ratio_min}, {args.low_ratio_max}] & "
        f"component_count>={args.low_ratio_component_count_threshold}",
        flush=True,
    )
    for parquet_path, requested in sorted(grouped.items()):
        print(
            f"[evaluate] parquet={Path(parquet_path).name} requested_rows={len(requested)} "
            f"checked_so_far={stats['checked']}",
            flush=True,
        )
        for requested_record, row in iter_requested_rows(parquet_path, requested, args.batch_size):
            stats["checked"] += 1
            reason = build_prefilter_reason(row, bit=args.bit, max_edge=args.max_edge)
            quality_metrics = None
            if reason is None:
                reason = validate_voxel_inputs(row)
            if reason is None:
                reason, quality_metrics = build_quality_reason(
                    row,
                    resolution=args.quality_resolution,
                    max_voxel_components=args.max_voxel_components,
                    max_small_component_ratio=args.max_small_component_ratio,
                    low_ratio_min=args.low_ratio_min,
                    low_ratio_max=args.low_ratio_max,
                    low_ratio_component_count_threshold=args.low_ratio_component_count_threshold,
                )
            if reason is None:
                stats["kept"] += 1
                continue

            removed_record = dict(requested_record)
            removed_record["reason"] = reason
            if quality_metrics is not None:
                removed_record["quality_metrics"] = quality_metrics
            removed.append(removed_record)
            stats["removed"] += 1
            stats[f"reason_{reason}"] += 1

            if args.progress_interval > 0 and stats["checked"] % args.progress_interval == 0:
                print_progress("evaluate", stats["checked"], total_records, start_time)

        if args.progress_interval > 0 and stats["checked"] % args.progress_interval != 0:
            print_progress("evaluate", stats["checked"], total_records, start_time)

    return removed, stats


def write_final_records(path: Path, records: Iterable[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    for record in records:
        write_jsonl(tmp_path, record)
    tmp_path.replace(path)


def remove_files(
    dataset_root: Path,
    removed_records: Sequence[dict],
    delete_cache: bool,
    progress_interval: int,
) -> Counter:
    stats: Counter = Counter()
    total_records = len(removed_records)
    start_time = time.time()
    print(
        f"[delete] start total_records={total_records} delete_cache={delete_cache}",
        flush=True,
    )
    for idx, record in enumerate(removed_records, start=1):
        split_path = dataset_root / record["split"] / "data" / f"{record['stem']}.pt"
        if split_path.exists():
            split_path.unlink()
            stats["dataset_files_removed"] += 1
        else:
            stats["dataset_files_missing"] += 1

        cache_path = record.get("cache_path")
        if delete_cache and cache_path:
            cache_file = Path(cache_path)
            if cache_file.exists():
                cache_file.unlink()
                stats["cache_files_removed"] += 1
            else:
                stats["cache_files_missing"] += 1

        if idx == total_records or (progress_interval > 0 and idx % progress_interval == 0):
            print_progress("delete", idx, total_records, start_time)
    return stats


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    final_records_path = dataset_root / "final_records.jsonl"
    train_metadata_path = dataset_root / "train" / "metadata.csv"
    val_metadata_path = dataset_root / "val" / "metadata.csv"

    manifest = load_json(manifest_path)
    selected_manifest_path = Path(manifest["source_manifest"])
    final_records = load_final_records(final_records_path)
    current_records = filter_current_records(final_records, args.only_stems)
    if not current_records:
        print("No records matched the requested scope.")
        return

    target_ranks = {int(record["rank"]) for record in current_records}
    selected_by_rank = {
        int(record["rank"]): record
        for record in load_jsonl(selected_manifest_path)
        if int(record["rank"]) in target_ranks
    }

    print(
        f"[main] dataset_root={dataset_root} current_records={len(current_records)} "
        f"selected_records={len(selected_by_rank)} dry_run={args.dry_run}",
        flush=True,
    )

    removed_records, eval_stats = evaluate_records(current_records, selected_by_rank, args)
    removed_ranks = {int(record["rank"]) for record in removed_records}
    kept_records = [record for record in final_records if int(record["rank"]) not in removed_ranks]
    kept_records.sort(key=lambda item: (split_order(item.get("split", "")), int(item["rank"])))

    train_records = [record for record in kept_records if record["split"] == "train"]
    val_records = [record for record in kept_records if record["split"] == "val"]

    report_dir = dataset_root / "_pipeline"
    report_dir.mkdir(parents=True, exist_ok=True)
    removed_jsonl_path = report_dir / "pruned_filtered_samples.jsonl"
    report_json_path = report_dir / "pruned_filtered_samples_report.json"

    if removed_jsonl_path.exists():
        removed_jsonl_path.unlink()
    for record in removed_records:
        write_jsonl(removed_jsonl_path, record)

    report = {
        "dataset_root": str(dataset_root),
        "checked_count": eval_stats["checked"],
        "removed_count": eval_stats["removed"],
        "kept_count": eval_stats["kept"],
        "only_stems": list(args.only_stems),
        "dry_run": bool(args.dry_run),
        "delete_cache": bool(args.delete_cache),
        "filter_config": {
            "bit": args.bit,
            "max_edge": args.max_edge,
            "quality_resolution": args.quality_resolution,
            "max_voxel_components": args.max_voxel_components,
            "max_small_component_ratio": args.max_small_component_ratio,
            "low_ratio_min": args.low_ratio_min,
            "low_ratio_max": args.low_ratio_max,
            "low_ratio_component_count_threshold": args.low_ratio_component_count_threshold,
            "progress_interval": args.progress_interval,
            "delete_progress_interval": args.delete_progress_interval,
        },
        "reason_stats": {
            key.replace("reason_", ""): value
            for key, value in sorted(eval_stats.items())
            if key.startswith("reason_")
        },
        "result_counts": {
            "target_count": len(kept_records),
            "val_count": len(val_records),
            "train_count": len(train_records),
        },
    }
    write_json(report_json_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.dry_run:
        print(f"Dry-run only. No files were deleted. Report: {report_json_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = dataset_root / "_pipeline" / "prune_backups" / timestamp
    backup_files(
        backup_dir,
        [manifest_path, final_records_path, train_metadata_path, val_metadata_path],
    )

    remove_stats = remove_files(
        dataset_root,
        removed_records,
        delete_cache=args.delete_cache,
        progress_interval=args.delete_progress_interval,
    )

    write_metadata(train_metadata_path, [record["stem"] for record in train_records])
    write_metadata(val_metadata_path, [record["stem"] for record in val_records])
    write_final_records(final_records_path, kept_records)

    manifest["target_count"] = len(kept_records)
    manifest["val_count"] = len(val_records)
    manifest["train_count"] = len(train_records)
    manifest["pruned_in_place"] = True
    manifest["prune_timestamp"] = timestamp
    manifest["prune_removed_count"] = len(removed_records)
    manifest["prune_filter_config"] = report["filter_config"]
    write_json(manifest_path, manifest)

    print(json.dumps(dict(remove_stats), indent=2, ensure_ascii=False))
    print(f"Backup written to {backup_dir}")


if __name__ == "__main__":
    main()