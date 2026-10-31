"""Create byte-traceable, benchmark-complete R1 holdout partitions without training."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from local_evaluation import ResponseRecord, load_records, sha256_file
from train_coldstart import split_by_benchmark, write_json, write_records

BENCHMARKS = ("researchcodebench", "real_webagents")
SOURCE_PATHS = (
    Path(__file__).resolve(),
    TOOLS_DIR / "train_coldstart.py",
    TOOLS_DIR / "local_evaluation.py",
    PROJECT_ROOT / "coldstart" / "features.py",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


# 校验用户提供的输入哈希格式，避免把不完整或非 SHA-256 值写入复现清单。
def parse_sha256(value: str) -> str:
    normalized = value.lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise argparse.ArgumentTypeError(
            "--records-sha256 must be 64 hexadecimal characters"
        )
    return normalized


# 确认输出根目录位于项目内且尚不存在，必须在读取任何输入前调用。
def validate_new_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Output root must be inside project root: {resolved}"
        ) from exc
    if resolved == PROJECT_ROOT:
        raise ValueError("Output root must be a new directory below the project root.")
    if resolved.exists():
        raise FileExistsError(f"Output root must be new and must not exist: {resolved}")
    return resolved


# 以固定闭包计算源码哈希，绑定分区入口及其直接复用的读写和特征依赖。
def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in SOURCE_PATHS
    }


# 检查 R1 记录只含两个规定 benchmark，防止混入其他数据后仍生成看似有效的留出集。
def validate_r1_benchmarks(records: list[ResponseRecord]) -> None:
    observed = {record.benchmark_id for record in records}
    expected = set(BENCHMARKS)
    if observed != expected:
        raise ValueError(
            "R1 records must contain exactly researchcodebench and real_webagents; "
            f"observed {sorted(observed)}."
        )


# 为一个完整 benchmark 方向写入唯一留出 JSONL 和历史兼容的四字段 split 清单。
def write_direction(
    output_root: Path, benchmark: str, records: list[ResponseRecord]
) -> dict[str, Any]:
    _, heldout_records = split_by_benchmark(records, {benchmark})
    direction_root = output_root / benchmark
    direction_root.mkdir()
    heldout_path = direction_root / "heldout.jsonl"
    split_path = direction_root / "split.json"
    write_records(heldout_path, heldout_records)
    split = {
        "heldout_benchmarks": [benchmark],
        "train_benchmarks": sorted(
            {
                record.benchmark_id
                for record in records
                if record.benchmark_id != benchmark
            }
        ),
        "heldout_records": len(heldout_records),
        "train_records": len(records) - len(heldout_records),
    }
    write_json(split_path, split)
    return {
        "benchmark": benchmark,
        "directory": benchmark,
        "heldout_records": len(heldout_records),
        "split": split,
        "outputs": {
            "heldout.jsonl": {
                "sha256": sha256_file(heldout_path),
                "bytes": heldout_path.stat().st_size,
            },
            "split.json": {
                "sha256": sha256_file(split_path),
                "bytes": split_path.stat().st_size,
            },
        },
    }


# 执行一次受哈希约束的双方向纯分区，并返回写入 reproduction 清单的追溯信息。
def reproduce_partitions(
    records_path: Path, expected_records_sha256: str, output_root: Path
) -> dict[str, Any]:
    output_root = validate_new_output_root(output_root)
    actual_records_sha256 = sha256_file(records_path)
    if actual_records_sha256 != expected_records_sha256:
        raise ValueError(
            "Input records SHA-256 mismatch: "
            f"expected {expected_records_sha256}, got {actual_records_sha256}."
        )
    records = load_records(records_path)
    validate_r1_benchmarks(records)
    source_sha256 = source_hashes()

    output_root.mkdir(parents=True)
    directions = {
        benchmark: write_direction(output_root, benchmark, records)
        for benchmark in BENCHMARKS
    }
    reproduction = {
        "tool": "tools/reproduce_r1_partitions.py",
        "purpose": "R1 complete-benchmark holdout partition reproduction only",
        "input": {
            "records_path": str(records_path.resolve()),
            "records_sha256": actual_records_sha256,
            "records_count": len(records),
        },
        "source_sha256": source_sha256,
        "directions": directions,
        "seed": {
            "value": None,
            "controls_membership": False,
            "note": "Membership is determined only by complete benchmark_id.",
        },
        "training": {
            "performed": False,
            "note": "No train, fit, score, worker, or network operation is called.",
        },
        "output_root": str(output_root),
    }
    write_json(output_root / "reproduction.json", reproduction)
    return reproduction


# 解析显式输入身份和新输出目录参数，保持入口不接受覆盖或隐式默认数据。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, help="Authorized R1 JSONL records"
    )
    parser.add_argument(
        "--records-sha256",
        type=parse_sha256,
        required=True,
        help="Required SHA-256 of --records",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New project-internal directory for partition artifacts; overwrite is unsupported",
    )
    return parser.parse_args(argv)


# 提供 CLI 入口，仅打印完成路径而不触发训练、评分或任何网络访问。
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reproduce_partitions(args.records, args.records_sha256, args.output_root)
    print(f"Wrote R1 benchmark partitions to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
