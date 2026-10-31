"""Run bounded training-side κ selection for raw-response q-train smoothers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
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
from tools.local_evaluation import (
    ALC_WEIGHTS,
    BUDGETS,
    ResponseRecord,
    group_records,
    group_seed,
    load_records,
    split_items,
)

SEED = 20261014
MIN_ITEMS = 80
CAP = 300
GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SOURCES = {
    "researchcodebench": ROOT
    / "runs/pilot_r1_holdout_researchcodebench/training/heldout.jsonl",
    "real_webagents": ROOT / "runs/pilot_r1_holdout_real/training/heldout.jsonl",
    "statqa": ROOT / "payload/prepared/r2_statqa/records.jsonl",
}
EXPECTED = {
    "researchcodebench": "53394dc05debf69096ec3210e0570f54becb432b3405f815250e66a43fde067b",
    "real_webagents": "c537a5c1297303c088ddc4e7dff1991f0b5b3275d31313f104074445fbd2d1c2",
    "statqa": "9302059479242986f3fbd56d1fb30b269828b967800ef97259008f270b8302e6",
}
MULTI = ROOT / "runs/20261014_multibench_study"


# 流式计算输入、源码与输出文件哈希，避免为身份记录整块载入 JSONL。
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


# 原子写入 JSON，保证超时和失败时保留可解析的阶段证据。
def dump(path: Path, value: dict[str, Any]) -> None:
    t = path.with_suffix(path.suffix + ".tmp")
    t.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(t, path)


# 记录 UTC 发生时间，不将其用于随机性或评分计算。
def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 返回运行时库版本，仅为复现身份记录而不改变环境。
def versions() -> dict[str, str]:
    import numpy
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
    }


# 计算 raw-response 训练成功率；不进行 canonical 去重或跨调用缓存。
def raw_q(records: list[ResponseRecord]) -> float:
    return sum(r.label for r in records) / len(records)


# 计算指定 q、κ 和当前已揭示标签上的 Beta 平滑概率。
def beta_probability(successes: int, count: int, q: float, kappa: float) -> float:
    if not (0.0 <= q <= 1.0 and kappa > 0 and 0 <= successes <= count):
        raise ValueError("invalid beta parameters")
    return (successes + kappa * q) / (count + kappa)


# 复刻局部协议的 item 切分、随机获取与 first_by_response_id 标签前缀，保持所有 response 在同 item 侧。
def group_prefixes(
    key: tuple[str, str], records: list[ResponseRecord]
) -> tuple[list[ResponseRecord], dict[int, tuple[int, int, bool]]]:
    acquisition, evaluation, by_item = split_items(records, SEED, key)
    evaluation_records = [r for item in evaluation for r in by_item[item]]
    rng = random.Random(group_seed(SEED + 1, key))
    candidates = list(acquisition)
    rng.shuffle(candidates)
    acquired: list[ResponseRecord] = []
    checkpoints: dict[int, tuple[int, int, bool]] = {0: (0, 0, True)}
    targets = set(BUDGETS[1:])
    for index, item_id in enumerate(candidates):
        if len(acquired) >= max(BUDGETS):
            break
        remaining = len(candidates) - index
        needed = max(BUDGETS) - len(acquired)
        if rng.random() < min(1.0, needed / remaining):
            acquired.append(min(by_item[item_id], key=lambda r: r.response_id))
            if len(acquired) in targets:
                checkpoints[len(acquired)] = (
                    sum(r.label for r in acquired),
                    len(acquired),
                    True,
                )
    for budget in BUDGETS[1:]:
        if budget not in checkpoints:
            checkpoints[budget] = (sum(r.label for r in acquired), len(acquired), False)
    return evaluation_records, checkpoints


# 对一个内验证 benchmark 的所有合格组一次计算七条 κ 曲线，组间等权且保留逐组 response-level Brier。
def benchmark_curves(records: list[ResponseRecord], q: float) -> dict[str, Any]:
    groups = group_records(records)
    curves = []
    complete_by_k: dict[float, list[float]] = {k: [] for k in GRID}
    for key, values in sorted(groups.items()):
        if len({r.item_id for r in values}) < MIN_ITEMS:
            continue
        evaluation, checkpoints = group_prefixes(key, values)
        per_k: dict[str, Any] = {}
        for kappa in GRID:
            budget_brier: dict[str, float | None] = {}
            all_complete = True
            for budget in BUDGETS:
                successes, count, complete = checkpoints[budget]
                if not complete:
                    budget_brier[str(budget)] = None
                    all_complete = False
                    continue
                p = beta_probability(successes, count, q, kappa)
                budget_brier[str(budget)] = sum(
                    (p - r.label) ** 2 for r in evaluation
                ) / len(evaluation)
            alc = (
                sum(
                    w * float(budget_brier[str(b)])
                    for b, w in zip(BUDGETS, ALC_WEIGHTS)
                )
                if all_complete
                else None
            )
            per_k[str(kappa)] = {"per_budget_brier": budget_brier, "alc": alc}
            if alc is not None:
                complete_by_k[kappa].append(alc)
        curves.append(
            {
                "group_id": hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[
                    :16
                ],
                "benchmark_id": key[1],
                "distinct_items": len({r.item_id for r in values}),
                "evaluation_responses": len(evaluation),
                "curves": per_k,
            }
        )
    losses = {str(k): sum(v) / len(v) if v else None for k, v in complete_by_k.items()}
    return {
        "q_from_other_inner_benchmark": q,
        "groups_eligible": len(curves),
        "groups_complete_by_kappa": {str(k): len(v) for k, v in complete_by_k.items()},
        "group_equal_alc_by_kappa": losses,
        "groups": curves,
    }


# 读取外折的两个训练任务，以 LOBO 交叉验证各自 raw q，并按两个内 benchmark 等权选择 κ。
def selection_worker(heldout: str, output: Path) -> int:
    train_names = [n for n in SOURCES if n != heldout]
    data = {n: load_records(SOURCES[n]) for n in train_names}
    for name, records in data.items():
        if {r.benchmark_id for r in records} != {name}:
            raise ValueError(f"unexpected benchmark in {name}")
    outer_records = data[train_names[0]] + data[train_names[1]]
    outer_q = raw_q(outer_records)
    multi_config = json.loads(
        (MULTI / "folds" / heldout / "config.json").read_text(encoding="utf-8")
    )
    multi_split = json.loads(
        (MULTI / "folds" / heldout / "split.json").read_text(encoding="utf-8")
    )
    if set(multi_split["train_benchmarks"]) != set(train_names) or multi_split[
        "heldout_benchmarks"
    ] != [heldout]:
        raise ValueError("multibench split does not match outer fold")
    if not math.isclose(outer_q, float(multi_config["q_train"]), rel_tol=0, abs_tol=0):
        raise ValueError("raw outer q differs from completed multibench record")
    inner = {
        validation: benchmark_curves(data[validation], raw_q(data[other]))
        for validation, other in (
            (train_names[0], train_names[1]),
            (train_names[1], train_names[0]),
        )
    }
    losses: dict[float, float] = {}
    for kappa in GRID:
        values = [
            inner[name]["group_equal_alc_by_kappa"][str(kappa)] for name in train_names
        ]
        if any(v is None for v in values):
            raise ValueError("inner validation has no complete groups")
        losses[kappa] = sum(float(v) for v in values) / len(values)
    selected = min(GRID, key=lambda k: (losses[k], -k))
    dump(
        output / "inner_curves.json",
        {
            "heldout_outer_benchmark": heldout,
            "seed": SEED,
            "min_distinct_items": MIN_ITEMS,
            "protocol": {
                "labeled_visibility": "target_group_only",
                "split": "item_50_50",
                "acquisition_response_selection": "first_by_response_id",
                "unit": "response-level Brier within group; group equal then inner-benchmark equal",
            },
            "inner": inner,
        },
    )
    dump(
        output / "selection.json",
        {
            "heldout_outer_benchmark": heldout,
            "outer_train_sources": train_names,
            "outer_raw_q": outer_q,
            "raw_response_counts": {n: len(data[n]) for n in train_names},
            "inner_kappa_loss_by_kappa": {str(k): losses[k] for k in GRID},
            "selected_kappa": selected,
            "tie_break": "minimum ALC; exact tie chooses larger kappa",
            "multibench_config_sha256": sha(MULTI / "folds" / heldout / "config.json"),
            "multibench_split_sha256": sha(MULTI / "folds" / heldout / "split.json"),
        },
    )
    return 0


# 将固定平滑模板复制为给定 raw q 与 κ 的单文件预测器，不读取 artifact 或标签状态。
def write_model(path: Path, q: float, kappa: float) -> None:
    text = (ROOT / "smoothed_mean/model.py").read_text(encoding="utf-8")
    if text.count("PRIOR_MEAN = 0.5") != 1 or text.count("PRIOR_STRENGTH = 2.0") != 1:
        raise ValueError("unexpected smoother template")
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        text.replace("PRIOR_MEAN = 0.5", f"PRIOR_MEAN = {q!r}").replace(
            "PRIOR_STRENGTH = 2.0", f"PRIOR_STRENGTH = {kappa!r}"
        ),
        encoding="utf-8",
        newline="\n",
    )


# 在共享 deadline 内运行一个 worker、评测或分析进程，超时终止后保存日志和阶段状态。
def stage(
    owner: Path, m: dict[str, Any], name: str, cmd: list[str], deadline: float
) -> None:
    left = deadline - time.monotonic()
    entry = {
        "name": name,
        "command": cmd,
        "remaining_seconds_at_start": max(0.0, left),
        "status": "planned",
    }
    m["stages"].append(entry)
    if left <= 0:
        entry["status"] = m["status"] = "timed_out"
        dump(owner / "manifest.json", m)
        raise TimeoutError(name)
    log = owner / "logs" / f"{len(m['stages']):02d}_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        entry.update(
            {"pid": p.pid, "status": "running", "log_path": str(log.relative_to(owner))}
        )
        m["status"] = "running"
        dump(owner / "manifest.json", m)
        timed = False
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed = True
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)
    entry.update(
        {
            "returncode": p.returncode,
            "seconds": time.monotonic() - started,
            "log_sha256": sha(log),
            "status": "timed_out"
            if timed
            else ("completed" if p.returncode == 0 else "failed"),
        }
    )
    if entry["status"] != "completed":
        m["status"] = entry["status"]
    dump(owner / "manifest.json", m)
    if timed:
        raise TimeoutError(name)
    if p.returncode:
        raise RuntimeError(name)


# 对一个冻结 q、κ 参数运行原 local evaluator 和独立 analyzer，并保存每格参数哈希。
def evaluate_cell(
    fold: Path,
    heldout: str,
    name: str,
    model: Path,
    q: float,
    kappa: float,
    deadline: float,
    command: list[str],
) -> dict[str, Any]:
    cell = fold / "cells" / name
    cell.mkdir(parents=True, exist_ok=False)
    m = {
        "status": "planned",
        "heldout_benchmark": heldout,
        "predictor": name,
        "seed": SEED,
        "min_distinct_items": MIN_ITEMS,
        "budgets": list(BUDGETS),
        "records_path": str(SOURCES[heldout]),
        "records_sha256": EXPECTED[heldout],
        "frozen_parameters": {"raw_q": q, "kappa": kappa, "model_sha256": sha(model)},
        "batch_command": command,
        "stages": [],
    }
    dump(cell / "manifest.json", m)
    out = cell / "evaluations" / name
    stage(
        cell,
        m,
        "evaluate",
        [
            sys.executable,
            "-B",
            str(ROOT / "tools/local_evaluation.py"),
            "--records",
            str(SOURCES[heldout]),
            "--model",
            str(model),
            "--output-dir",
            str(out),
            "--seed",
            str(SEED),
            "--min-distinct-items",
            str(MIN_ITEMS),
        ],
        deadline,
    )
    stage(
        cell,
        m,
        "independent_analysis",
        [
            sys.executable,
            "-B",
            str(ROOT / "tools/analyze_development_batch.py"),
            "--batch-dir",
            str(cell),
            "--output-dir",
            str(cell / "analysis"),
        ],
        deadline,
    )
    x = json.loads((cell / "analysis/comparison.json").read_text(encoding="utf-8"))[
        "models"
    ][0]
    result = {
        "predictor": name,
        "heldout_benchmark": heldout,
        "raw_q": q,
        "kappa": kappa,
        "b0": x["per_budget"]["0"]["mean_brier"],
        "b1": x["per_budget"]["1"]["mean_brier"],
        "b3": x["per_budget"]["3"]["mean_brier"],
        "b31": x["per_budget"]["31"]["mean_brier"],
        "alc": x["mean_brier_alc_common_complete_groups"],
        "complete_groups": x["groups_complete_all_budgets"],
        "missing_budget_entries": len(x["missing_budget_entries"]),
        "summary_mismatches": len(x["summary_mismatches"]),
    }
    m.update({"status": "completed", "analysis": result})
    dump(cell / "manifest.json", m)
    return result


# 执行一个外折的可终止内层选择，再固定 selected κ 与固定 κ=2 两个外层对照。
def run_fold(
    root: Path, heldout: str, deadline: float, command: list[str]
) -> list[dict[str, Any]]:
    fold = root / "folds" / heldout
    fold.mkdir(parents=True, exist_ok=False)
    m = {
        "status": "planned",
        "heldout_benchmark": heldout,
        "outer_train_benchmarks": [n for n in SOURCES if n != heldout],
        "input_sha256": EXPECTED,
        "grid": list(GRID),
        "seed": SEED,
        "min_distinct_items": MIN_ITEMS,
        "stages": [],
    }
    dump(fold / "manifest.json", m)
    stage(
        fold,
        m,
        "inner_kappa_selection",
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--selection-worker",
            "--heldout-benchmark",
            heldout,
            "--output-root",
            str(fold),
        ],
        deadline,
    )
    selected = json.loads((fold / "selection.json").read_text(encoding="utf-8"))
    q = float(selected["outer_raw_q"])
    k = float(selected["selected_kappa"])
    models = fold / "predictors"
    write_model(models / "selected_kappa/model.py", q, k)
    write_model(models / "fixed_kappa2/model.py", q, 2.0)
    frozen = {
        n: {"raw_q": q, "kappa": kk, "model_sha256": sha(models / f"{n}/model.py")}
        for n, kk in (("selected_kappa", k), ("fixed_kappa2", 2.0))
    }
    m.update(
        {
            "status": "selected",
            "selection_sha256": sha(fold / "selection.json"),
            "inner_curves_sha256": sha(fold / "inner_curves.json"),
            "frozen_parameters": frozen,
        }
    )
    dump(fold / "manifest.json", m)
    results = [
        evaluate_cell(
            fold, heldout, n, models / f"{n}/model.py", q, kk, deadline, command
        )
        for n, kk in (("selected_kappa", k), ("fixed_kappa2", 2.0))
    ]
    m.update({"status": "completed"})
    dump(fold / "manifest.json", m)
    return results


# 执行无需真实数据加载的公式与 outer-exclusion 小检查，防止基本 κ 语义或选择泄漏。
def self_check() -> int:
    assert beta_probability(1, 2, 0.25, 2.0) == 0.375
    for outer in SOURCES:
        assert outer not in [n for n in SOURCES if n != outer]
    print("kappa formula and outer-exclusion check passed")
    return 0


# 快照实际执行源码和关键 helper，避免只记录可变工作树路径。
def snapshot(root: Path) -> dict[str, dict[str, str]]:
    d = root / "source"
    d.mkdir(parents=True, exist_ok=False)
    paths = {
        "runner.py": Path(__file__).resolve(),
        "local_evaluation.py": ROOT / "tools/local_evaluation.py",
        "analyze_development_batch.py": ROOT / "tools/analyze_development_batch.py",
        "smoothed_mean_model.py": ROOT / "smoothed_mean/model.py",
        "multibench_plan.json": MULTI / "plan.json",
    }
    out = {}
    for name, p in paths.items():
        dest = d / name
        shutil.copy2(p, dest)
        out[name] = {
            "source_path": str(p),
            "source_sha256": sha(p),
            "snapshot_sha256": sha(dest),
        }
        if out[name]["source_sha256"] != out[name]["snapshot_sha256"]:
            raise RuntimeError("snapshot mismatch")
    return out


# 验证完成的 multibench 基线输入身份与三个外折 raw q 来源，拒绝漂移后重用。
def validate_multibench() -> dict[str, Any]:
    plan = json.loads((MULTI / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((MULTI / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("multibench source not completed")
    for n, source in SOURCES.items():
        if (
            plan["input_identity"][n]["sha256"] != EXPECTED[n]
            or sha(source) != EXPECTED[n]
        ):
            raise ValueError(f"input source mismatch {n}")
    return {
        "plan_sha256": sha(MULTI / "plan.json"),
        "summary_sha256": sha(MULTI / "summary.json"),
        "summary_seconds": summary["seconds"],
    }


# 解析正常批次、内层选择 worker 和小检查入口，正常批次仅接受全新的输出目录。
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path)
    p.add_argument("--cap-seconds", type=int, default=CAP)
    p.add_argument("--selection-worker", action="store_true")
    p.add_argument("--heldout-benchmark", choices=tuple(SOURCES))
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()
    if a.selection_worker:
        if a.output_root is None or a.heldout_benchmark is None:
            p.error("selection worker requires output and heldout")
        return selection_worker(a.heldout_benchmark, a.output_root.resolve())
    if a.output_root is None:
        p.error("output root required")
    if a.cap_seconds <= 0:
        p.error("cap must be positive")
    out = a.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + a.cap_seconds
    cmd = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--output-root",
        str(out),
        "--cap-seconds",
        str(a.cap_seconds),
    ]
    plan = {
        "status": "planned",
        "purpose": "raw_response_q_train_kappa_training_side_selection",
        "created_at_utc": now(),
        "command": cmd,
        "cap_seconds": a.cap_seconds,
        "cost_cny": 0,
        "seed": SEED,
        "grid": list(GRID),
        "parameters": {
            "prior_formula": "(successes + kappa * raw_q_train) / (labels + kappa)",
            "raw_q_definition": "all raw response labels, no canonical deduplication",
            "min_distinct_items": MIN_ITEMS,
            "budgets": list(BUDGETS),
            "tie_break": "minimum ALC then larger kappa",
        },
        "outer_folds": [
            {"heldout": h, "train": [n for n in SOURCES if n != h]} for h in SOURCES
        ],
        "input_sha256": {
            n: {"path": str(SOURCES[n]), "sha256": EXPECTED[n]} for n in SOURCES
        },
        "multibench_source": validate_multibench(),
        "runtime_versions": versions(),
        "source_snapshot": snapshot(out),
        "protocol_limitations": [
            "local target_group_only and first_by_response_id controlled protocol",
            "official response reveal ordering, cross-group timing, and private max=1000 sampling not verified",
            "not an official all-revealed protocol or platform improvement claim",
        ],
    }
    dump(out / "plan.json", plan)
    results = []
    planned = [
        {"heldout_benchmark": h, "predictor": n}
        for h in SOURCES
        for n in ("selected_kappa", "fixed_kappa2")
    ]
    try:
        for heldout in SOURCES:
            results.extend(run_fold(out, heldout, deadline, cmd))
        by = {}
        for r in results:
            by.setdefault(r["heldout_benchmark"], {})[r["predictor"]] = r
        deltas = [
            by[h]["selected_kappa"]["alc"] - by[h]["fixed_kappa2"]["alc"]
            for h in SOURCES
        ]
        criterion = sum(d < -0.001 for d in deltas) >= 2 and all(
            d <= 0.001 for d in deltas
        )
        dump(
            out / "summary.json",
            {
                "status": "completed",
                "seconds": time.monotonic() - started,
                "cap_seconds": a.cap_seconds,
                "cost_cny": 0,
                "results": results,
                "selected_minus_fixed_kappa2_alc": {
                    h: by[h]["selected_kappa"]["alc"] - by[h]["fixed_kappa2"]["alc"]
                    for h in SOURCES
                },
                "predeclared_followup_condition": {
                    "condition": "at least 2/3 tasks lower ALC by >0.001 and no task worse by >0.001",
                    "met": criterion,
                },
                "not_started": [],
                "limitations": plan["protocol_limitations"],
            },
        )
        return 0
    except Exception as e:
        complete = {(r["heldout_benchmark"], r["predictor"]) for r in results}
        dump(
            out / "summary.json",
            {
                "status": "timed_out" if isinstance(e, TimeoutError) else "failed",
                "error_type": type(e).__name__,
                "seconds": time.monotonic() - started,
                "cap_seconds": a.cap_seconds,
                "cost_cny": 0,
                "results": results,
                "not_started": [
                    x
                    for x in planned
                    if (x["heldout_benchmark"], x["predictor"]) not in complete
                ],
                "limitations": plan["protocol_limitations"],
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
