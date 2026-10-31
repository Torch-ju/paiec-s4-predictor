"""Run a bounded three-benchmark leave-one-benchmark-out development study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_evaluation import ResponseRecord, load_records
from tools.train_coldstart import fit_candidate, make_artifact

SEED = 20261014
MAX_FEATURES = 20_000
PRIOR_STRENGTH = 2.0
MIN_DISTINCT_ITEMS = 80
CAP_SECONDS = 300
SOURCES = {
    "researchcodebench": ROOT
    / "runs/pilot_r1_holdout_researchcodebench/training/heldout.jsonl",
    "real_webagents": ROOT / "runs/pilot_r1_holdout_real/training/heldout.jsonl",
    "statqa": ROOT / "payload/prepared/r2_statqa/records.jsonl",
}
EXPECTED_SHA256 = {
    "researchcodebench": "53394dc05debf69096ec3210e0570f54becb432b3405f815250e66a43fde067b",
    "real_webagents": "c537a5c1297303c088ddc4e7dff1991f0b5b3275d31313f104074445fbd2d1c2",
    "statqa": "9302059479242986f3fbd56d1fb30b269828b967800ef97259008f270b8302e6",
}


# 流式计算文件字节身份，避免为记录 SHA 而一次性载入大 JSONL。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 原子写入 JSON 阶段记录，保留失败和超时发生前已获得的证据。
def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# 返回 UTC 时间，清晰记录计划、进程和终态的实际发生顺序。
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 读取实际运行时版本，只记录环境身份且不安装或变更任何依赖。
def runtime_versions() -> dict[str, str]:
    import numpy
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
    }


# 验证三份固定输入存在且哈希匹配，阻止来源漂移后仍沿用原协议名称。
def validate_inputs() -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != EXPECTED_SHA256[name]:
            raise ValueError(f"input SHA mismatch for {name}")
        evidence[name] = {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return evidence


# 冻结执行与被导入 helper 源码副本，使批次可定位到真实未提交文件内容。
def snapshot_sources(output_root: Path) -> dict[str, dict[str, str]]:
    source_dir = output_root / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "runner.py": Path(__file__).resolve(),
        "train_coldstart.py": ROOT / "tools/train_coldstart.py",
        "coldstart_model.py": ROOT / "coldstart/model.py",
        "coldstart_features.py": ROOT / "coldstart/features.py",
        "smoothed_mean_model.py": ROOT / "smoothed_mean/model.py",
        "local_evaluation.py": ROOT / "tools/local_evaluation.py",
        "analyze_development_batch.py": ROOT / "tools/analyze_development_batch.py",
    }
    evidence: dict[str, dict[str, str]] = {}
    for saved_name, path in paths.items():
        destination = source_dir / saved_name
        shutil.copy2(path, destination)
        evidence[saved_name] = {
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "snapshot_sha256": sha256_file(destination),
        }
        if (
            evidence[saved_name]["source_sha256"]
            != evidence[saved_name]["snapshot_sha256"]
        ):
            raise RuntimeError(f"source snapshot mismatch: {path}")
    return evidence


# 仅替换冻结平滑均值源码的先验常数，生成训练侧 q 的无状态 κ=2 对照。
def write_q_train_model(destination: Path, q_train: float) -> None:
    template = (ROOT / "smoothed_mean/model.py").read_text(encoding="utf-8")
    old = "PRIOR_MEAN = 0.5"
    if template.count(old) != 1:
        raise ValueError("expected one smoothed-mean prior constant")
    destination.write_text(
        template.replace(old, f"PRIOR_MEAN = {q_train!r}"),
        encoding="utf-8",
        newline="\n",
    )


# 将两个训练来源分别读取后在内存合并，验证 benchmark 分割且绝不写联合 JSONL。
def load_training_records(
    heldout_name: str, training_names: list[str]
) -> tuple[list[ResponseRecord], dict[str, Any]]:
    records: list[ResponseRecord] = []
    source_counts: dict[str, int] = {}
    seen_response_keys: set[tuple[str, str]] = set()
    for name in training_names:
        source_records = load_records(SOURCES[name])
        actual_benchmarks = {record.benchmark_id for record in source_records}
        if actual_benchmarks != {name}:
            raise ValueError(
                f"{name} input has unexpected benchmarks: {actual_benchmarks}"
            )
        for record in source_records:
            key = (record.benchmark_id, record.response_id)
            if key in seen_response_keys:
                raise ValueError(f"duplicate train response key: {key}")
            seen_response_keys.add(key)
        source_counts[name] = len(source_records)
        records.extend(source_records)
    test_records = load_records(SOURCES[heldout_name])
    test_benchmarks = {record.benchmark_id for record in test_records}
    train_benchmarks = {record.benchmark_id for record in records}
    if test_benchmarks != {heldout_name} or heldout_name in train_benchmarks:
        raise ValueError("outer train/heldout benchmark sets are not disjoint")
    if len({record.label for record in records}) != 2:
        raise ValueError("outer training labels must contain both binary classes")
    return records, {
        "train_sources": training_names,
        "heldout_source": heldout_name,
        "train_source_counts": source_counts,
        "train_records": len(records),
        "heldout_records": len(test_records),
        "train_benchmarks": sorted(train_benchmarks),
        "heldout_benchmarks": sorted(test_benchmarks),
        "disjoint": True,
    }


# 在可终止 worker 中拟合完整文本 TF-IDF/liblinear，并冻结同训练集 q_train κ=2 对照。
def train_worker(heldout_name: str, output_dir: Path) -> int:
    training_names = [name for name in SOURCES if name != heldout_name]
    records, split = load_training_records(heldout_name, training_names)
    q_train = sum(record.label for record in records) / len(records)
    vectorizer, classifier = fit_candidate(records, SEED, MAX_FEATURES)
    coldstart_dir = output_dir / "predictors/coldstart"
    q_train_dir = output_dir / "predictors/q_train_smoothed"
    coldstart_dir.mkdir(parents=True, exist_ok=False)
    q_train_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "coldstart/model.py", coldstart_dir / "model.py")
    shutil.copy2(ROOT / "coldstart/features.py", coldstart_dir / "features.py")
    artifact = make_artifact(vectorizer, classifier, PRIOR_STRENGTH)
    write_json(coldstart_dir / "artifact.json", artifact)
    write_q_train_model(q_train_dir / "model.py", q_train)
    predictor_hashes = {
        "coldstart": {
            name: sha256_file(coldstart_dir / name)
            for name in ("model.py", "features.py", "artifact.json")
        },
        "q_train_smoothed": {"model.py": sha256_file(q_train_dir / "model.py")},
    }
    config = {
        "seed": SEED,
        "max_features": MAX_FEATURES,
        "classifier": {
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 300,
            "class_weight": None,
        },
        "prior_strength": PRIOR_STRENGTH,
        "q_train": q_train,
        "features": "coldstart.features.build_document full visible subject plus item_content/item_features/interactors",
        "no_combined_jsonl_written": True,
    }
    write_json(output_dir / "split.json", split)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "predictor_hashes.json", predictor_hashes)
    return 0


# 以共享总 deadline 运行单个子进程，超时后有限终止并保留 PID、日志与返回码。
def run_stage(
    owner: Path,
    manifest: dict[str, Any],
    name: str,
    command: list[str],
    deadline: float,
) -> None:
    remaining = deadline - time.monotonic()
    entry: dict[str, Any] = {
        "name": name,
        "command": command,
        "remaining_seconds_at_start": max(0.0, remaining),
        "status": "planned",
    }
    manifest["stages"].append(entry)
    if remaining <= 0:
        entry["status"] = "timed_out"
        manifest["status"] = "timed_out"
        write_json(owner / "manifest.json", manifest)
        raise TimeoutError(f"deadline expired before {name}")
    log_path = owner / "logs" / f"{len(manifest['stages']):02d}_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        entry.update(
            {
                "pid": process.pid,
                "status": "running",
                "log_path": str(log_path.relative_to(owner)),
            }
        )
        manifest["status"] = "running"
        write_json(owner / "manifest.json", manifest)
        timed_out = False
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    entry.update(
        {
            "returncode": process.returncode,
            "seconds": time.monotonic() - started,
            "log_sha256": sha256_file(log_path),
            "status": "timed_out"
            if timed_out
            else ("completed" if process.returncode == 0 else "failed"),
        }
    )
    if entry["status"] != "completed":
        manifest["status"] = entry["status"]
    write_json(owner / "manifest.json", manifest)
    if timed_out:
        raise TimeoutError(f"{name} exceeded total deadline")
    if process.returncode:
        raise RuntimeError(f"{name} exited {process.returncode}")


# 读取单模型独立复算输出，返回六档 Brier、共同完整分母与摘要冲突数。
def read_analysis(cell_dir: Path) -> dict[str, Any]:
    comparison = json.loads(
        (cell_dir / "analysis/comparison.json").read_text(encoding="utf-8")
    )
    models = comparison.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError("per-cell analyzer must report exactly one model")
    model = models[0]
    return {
        "b0": model["per_budget"]["0"]["mean_brier"],
        "alc": model["mean_brier_alc_common_complete_groups"],
        "b31": model["per_budget"]["31"]["mean_brier"],
        "groups_complete_all_budgets": model["groups_complete_all_budgets"],
        "groups_declared": model["groups_declared"],
        "missing_budget_entries": len(model["missing_budget_entries"]),
        "summary_mismatches": len(model["summary_mismatches"]),
    }


# 运行一个训练完成预测器的本地六档评测及其独立单模型复算。
def run_evaluation_cell(
    fold_dir: Path,
    heldout_name: str,
    predictor_name: str,
    deadline: float,
    cap_seconds: int,
    batch_command: list[str],
) -> dict[str, Any]:
    cell_dir = fold_dir / "cells" / predictor_name
    cell_dir.mkdir(parents=True, exist_ok=False)
    predictor_path = fold_dir / "predictors" / predictor_name / "model.py"
    manifest: dict[str, Any] = {
        "status": "planned",
        "heldout_benchmark": heldout_name,
        "predictor": predictor_name,
        "seed": SEED,
        "min_distinct_items": MIN_DISTINCT_ITEMS,
        "budgets": [0, 1, 3, 7, 15, 31],
        "cap_seconds": cap_seconds,
        "batch_command": batch_command,
        "records_path": str(SOURCES[heldout_name]),
        "records_sha256": EXPECTED_SHA256[heldout_name],
        "predictor_files_sha256": json.loads(
            (fold_dir / "predictor_hashes.json").read_text(encoding="utf-8")
        )[predictor_name],
        "training_config_sha256": sha256_file(fold_dir / "config.json"),
        "training_split_sha256": sha256_file(fold_dir / "split.json"),
        "stages": [],
    }
    write_json(cell_dir / "manifest.json", manifest)
    evaluation_dir = cell_dir / "evaluations" / predictor_name
    run_stage(
        cell_dir,
        manifest,
        "evaluate",
        [
            sys.executable,
            "-B",
            str(ROOT / "tools/local_evaluation.py"),
            "--records",
            str(SOURCES[heldout_name]),
            "--model",
            str(predictor_path),
            "--output-dir",
            str(evaluation_dir),
            "--seed",
            str(SEED),
            "--min-distinct-items",
            str(MIN_DISTINCT_ITEMS),
        ],
        deadline,
    )
    run_stage(
        cell_dir,
        manifest,
        "independent_analysis",
        [
            sys.executable,
            "-B",
            str(ROOT / "tools/analyze_development_batch.py"),
            "--batch-dir",
            str(cell_dir),
            "--output-dir",
            str(cell_dir / "analysis"),
        ],
        deadline,
    )
    analysis = read_analysis(cell_dir)
    manifest.update(
        {"status": "completed", "analysis": analysis, "completed_at_utc": utc_now()}
    )
    write_json(cell_dir / "manifest.json", manifest)
    return {"heldout_benchmark": heldout_name, "predictor": predictor_name, **analysis}


# 执行单个外折训练并依次评测冷启动与 q_train κ=2，任何终止均保留已完成单元。
def run_fold(
    output_root: Path,
    heldout_name: str,
    deadline: float,
    cap_seconds: int,
    batch_command: list[str],
) -> list[dict[str, Any]]:
    fold_dir = output_root / "folds" / heldout_name
    fold_dir.mkdir(parents=True, exist_ok=False)
    fold_manifest: dict[str, Any] = {
        "status": "planned",
        "heldout_benchmark": heldout_name,
        "train_benchmarks": [name for name in SOURCES if name != heldout_name],
        "input_sha256": {name: EXPECTED_SHA256[name] for name in SOURCES},
        "seed": SEED,
        "max_features": MAX_FEATURES,
        "prior_strength": PRIOR_STRENGTH,
        "min_distinct_items": MIN_DISTINCT_ITEMS,
        "stages": [],
    }
    write_json(fold_dir / "manifest.json", fold_manifest)
    run_stage(
        fold_dir,
        fold_manifest,
        "train",
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--train-worker",
            "--heldout-benchmark",
            heldout_name,
            "--output-root",
            str(fold_dir),
        ],
        deadline,
    )
    fold_manifest.update(
        {
            "status": "trained",
            "training_config_sha256": sha256_file(fold_dir / "config.json"),
            "training_split_sha256": sha256_file(fold_dir / "split.json"),
            "predictor_hashes_sha256": sha256_file(fold_dir / "predictor_hashes.json"),
            "predictor_files_sha256": json.loads(
                (fold_dir / "predictor_hashes.json").read_text(encoding="utf-8")
            ),
        }
    )
    write_json(fold_dir / "manifest.json", fold_manifest)
    results = []
    for predictor_name in ("coldstart", "q_train_smoothed"):
        results.append(
            run_evaluation_cell(
                fold_dir,
                heldout_name,
                predictor_name,
                deadline,
                cap_seconds,
                batch_command,
            )
        )
    fold_manifest.update({"status": "completed", "completed_at_utc": utc_now()})
    write_json(fold_dir / "manifest.json", fold_manifest)
    return results


# 对固定来源和 q 模板执行无需真实数据加载的最小源码解析检查。
def self_check() -> int:
    template = ROOT / "smoothed_mean/model.py"
    content = template.read_text(encoding="utf-8")
    if content.count("PRIOR_MEAN = 0.5") != 1:
        raise AssertionError("q-train template prior is not uniquely replaceable")
    if set(SOURCES) != set(EXPECTED_SHA256):
        raise AssertionError("source identity mapping is incomplete")
    print("multibench source-template check passed")
    return 0


# 解析受限入口并在新目录执行唯一共享 300 秒 CPU 批次。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cap-seconds", type=int, default=CAP_SECONDS)
    parser.add_argument("--train-worker", action="store_true")
    parser.add_argument("--heldout-benchmark", choices=tuple(SOURCES))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if args.train_worker:
        if args.heldout_benchmark is None or args.output_root is None:
            parser.error("train worker requires --heldout-benchmark and --output-dir")
        return train_worker(args.heldout_benchmark, args.output_root.resolve())
    if args.output_root is None:
        parser.error("--output-root is required")
    if args.cap_seconds <= 0:
        parser.error("--cap-seconds must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.cap_seconds
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--output-root",
        str(output_root),
        "--cap-seconds",
        str(args.cap_seconds),
    ]
    plan: dict[str, Any] = {
        "status": "planned",
        "purpose": "three_benchmark_leave_one_benchmark_out_development_study",
        "created_at_utc": utc_now(),
        "command": command,
        "cap_seconds": args.cap_seconds,
        "cost_cny": 0,
        "seed": SEED,
        "parameters": {
            "tfidf_max_features": MAX_FEATURES,
            "logistic_regression": {
                "solver": "liblinear",
                "C": 1.0,
                "max_iter": 300,
                "class_weight": None,
            },
            "prior_strength": PRIOR_STRENGTH,
            "min_distinct_items": MIN_DISTINCT_ITEMS,
            "budgets": [0, 1, 3, 7, 15, 31],
        },
        "outer_folds": [
            {"heldout": heldout, "train": [name for name in SOURCES if name != heldout]}
            for heldout in SOURCES
        ],
        "input_identity": validate_inputs(),
        "runtime_versions": runtime_versions(),
        "source_snapshot": snapshot_sources(output_root),
        "scope_limits": [
            "no combined JSONL written",
            "no parameter tuning",
            "no GPU, API, network download, or platform upload",
            "development comparison only; not a final independent model selection",
        ],
    }
    write_json(output_root / "plan.json", plan)
    results: list[dict[str, Any]] = []
    planned_cells = [
        {"heldout_benchmark": heldout, "predictor": predictor}
        for heldout in SOURCES
        for predictor in ("coldstart", "q_train_smoothed")
    ]
    try:
        for heldout in SOURCES:
            results.extend(
                run_fold(output_root, heldout, deadline, args.cap_seconds, command)
            )
        write_json(
            output_root / "summary.json",
            {
                "status": "completed",
                "seconds": time.monotonic() - started,
                "cap_seconds": args.cap_seconds,
                "cost_cny": 0,
                "results": results,
                "not_started": [],
                "purpose_limit": "tests whether adding one benchmark to training improves transfer; it does not select a final independent winner from these three development tasks.",
            },
        )
        return 0
    except Exception as error:
        completed = {(item["heldout_benchmark"], item["predictor"]) for item in results}
        not_started = [
            item
            for item in planned_cells
            if (item["heldout_benchmark"], item["predictor"]) not in completed
        ]
        write_json(
            output_root / "summary.json",
            {
                "status": "timed_out" if isinstance(error, TimeoutError) else "failed",
                "error_type": type(error).__name__,
                "seconds": time.monotonic() - started,
                "cap_seconds": args.cap_seconds,
                "cost_cny": 0,
                "results": results,
                "not_started": not_started,
                "purpose_limit": "tests whether adding one benchmark to training improves transfer; it does not select a final independent winner from these three development tasks.",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
