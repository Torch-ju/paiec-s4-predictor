"""Bounded visible-subject text probe with offline centered task nuisance effects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from feature_ablation.subject_only.features import build_document, feature_schema
from tools.local_evaluation import (
    ALC_WEIGHTS,
    BUDGETS,
    ResponseRecord,
    group_records,
    stable_json,
)
from tools.run_kappa_selection import group_prefixes
from tools.run_multibench_study import run_stage

SEED = 20261014
MIN_ITEMS = 80
CELLS = {
    "rawq_constant": ("rawq", None),
    "rawq_kappa2": ("rawq", 2.0),
    "nuisance_constant": ("nuisance", None),
    "nuisance_kappa2": ("nuisance", 2.0),
    "subject_nuisance_constant": ("subject_nuisance", None),
    "subject_nuisance_kappa2": ("subject_nuisance", 2.0),
}
SOURCES = {
    "researchcodebench": ROOT
    / "runs/pilot_r1_holdout_researchcodebench/training/heldout.jsonl",
    "real_webagents": ROOT / "runs/pilot_r1_holdout_real/training/heldout.jsonl",
    "statqa": ROOT / "payload/prepared/r2_statqa/records.jsonl",
    "truthfulqa_mc": ROOT / "payload/prepared/r3_truthfulqa/records.jsonl",
}
EXPECTED = {
    "researchcodebench": "53394dc05debf69096ec3210e0570f54becb432b3405f815250e66a43fde067b",
    "real_webagents": "c537a5c1297303c088ddc4e7dff1991f0b5b3275d31313f104074445fbd2d1c2",
    "statqa": "9302059479242986f3fbd56d1fb30b269828b967800ef97259008f270b8302e6",
    "truthfulqa_mc": "fb155648730d487687ef00b30854206261b35d920341e9def042c6aece8ee846",
}
DEPENDENCIES = [
    Path(__file__).resolve(),
    ROOT / "feature_ablation/subject_only/features.py",
    ROOT / "tools/local_evaluation.py",
    ROOT / "tools/run_kappa_selection.py",
    ROOT / "tools/run_multibench_study.py",
    ROOT / "tools/train_coldstart.py",
    ROOT / "coldstart/features.py",
]


# 流式记录字节身份，不把源码提交号替代实际执行文件。
def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 原子持久化配置、阶段结果及失败证据。
def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# 对任意恒定组概率计算全部响应的精确 Bernoulli Brier；重复响应计入 N/S。
def brier_from_counts(probability: float, successes: int, count: int) -> float:
    if not 0 <= probability <= 1 or not 0 <= successes <= count or count <= 0:
        raise ValueError("invalid probability or response counts")
    rate = successes / count
    return probability * probability - 2 * probability * rate + rate


# 单次流式读取仅保留 subject 与分组/响应标识，题目正文不进入本候选。
def read_compact(path: Path, benchmark: str) -> list[ResponseRecord]:
    records: list[ResponseRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if (
                raw["benchmark_id"] != benchmark
                or raw["item"].get("benchmark_id", benchmark) != benchmark
            ):
                raise ValueError("benchmark identity mismatch")
            if type(raw["label"]) is not int or raw["label"] not in (0, 1):
                raise ValueError("nonbinary response")
            if not isinstance(raw["subject"], dict) or any(
                not isinstance(v, str) for v in raw["subject"].values()
            ):
                raise ValueError("expected visible string subject fields")
            for field in ("response_id", "item_id"):
                if not isinstance(raw[field], str) or not raw[field]:
                    raise ValueError(f"missing {field}")
            if raw["response_id"] in seen:
                raise ValueError("duplicate response_id")
            seen.add(raw["response_id"])
            records.append(
                ResponseRecord(
                    raw["subject"],
                    {"benchmark_id": benchmark},
                    raw["label"],
                    raw["response_id"],
                    raw["item_id"],
                    benchmark,
                )
            )
    if not records:
        raise ValueError("empty data source")
    return records


# 按完整可见 subject 聚合响应均值和既有获取轨迹；每个响应保持原分组与预算侧。
def collect_groups(records: list[ResponseRecord]) -> list[dict[str, Any]]:
    result = []
    for key, values in sorted(group_records(records).items()):
        item_count = len({record.item_id for record in values})
        group = {
            "group_id": hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[:16],
            "benchmark": key[1],
            "subject": values[0].subject,
            "document": build_document(values[0].subject, {}),
            "responses": len(values),
            "successes": sum(record.label for record in values),
            "distinct_items": item_count,
            "eligible": item_count >= MIN_ITEMS,
        }
        if group["eligible"]:
            evaluation, checkpoints = group_prefixes(key, values)
            group.update(
                {
                    "evaluation_N": len(evaluation),
                    "evaluation_S": sum(record.label for record in evaluation),
                    "evaluation_response_ids_sha256": hashlib.sha256(
                        stable_json(sorted(r.response_id for r in evaluation)).encode()
                    ).hexdigest(),
                    "checkpoints": {
                        str(b): {
                            "s": checkpoints[b][0],
                            "n": checkpoints[b][1],
                            "complete": checkpoints[b][2],
                        }
                        for b in BUDGETS
                    },
                }
            )
        result.append(group)
    return result


# 两个 soft rows 严格等价组内全部响应加权损失，任务总权重均为 G/K。
def soft_counts(
    groups: list[dict[str, Any]], names: list[str]
) -> tuple[list[int], list[float], list[float]]:
    counts = Counter(g["benchmark"] for g in groups)
    if set(counts) != set(names):
        raise ValueError("training benchmark groups missing")
    labels, weights, group_weights = [], [], []
    for group in groups:
        weight = len(groups) / (len(names) * counts[group["benchmark"]])
        rate = group["successes"] / group["responses"]
        labels.extend([1, 0])
        weights.extend([rate * weight, (1 - rate) * weight])
        group_weights.append(weight)
    return labels, weights, group_weights


# 训练任务 contrast 零和中心化；推理时不见任何任务标识、全部置零。
def contrast(benchmarks: list[str], names: list[str]):
    import numpy as np

    return np.asarray(
        [
            [float(benchmark == name) - 1 / len(names) for name in names]
            for benchmark in benchmarks
        ]
    )


# 单次拟合两种同损失候选，保存实际 TF-IDF、系数和告警；留出标签不进入拟合。
def fit_priors(training: list[dict[str, Any]], names: list[str], output: Path):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    labels, weights, group_weights = soft_counts(training, names)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        norm="l2",
        smooth_idf=True,
        sublinear_tf=False,
        min_df=1,
        max_features=20000,
    )
    text = vectorizer.fit_transform([group["document"] for group in training])
    task = contrast([group["benchmark"] for group in training], names)
    indices = np.repeat(np.arange(len(training)), 2)
    matrices = {
        "nuisance": csr_matrix(task[indices]),
        "subject_nuisance": hstack([text[indices], task[indices]], format="csr"),
    }
    classifiers = {}
    metadata = {
        "train_benchmarks": names,
        "training_groups": len(training),
        "train_all_valid_groups_including_short": True,
        "sum_weights": float(sum(weights)),
        "group_weights": [
            {
                "group_id": g["group_id"],
                "benchmark": g["benchmark"],
                "responses": g["responses"],
                "successes": g["successes"],
                "weight": weight,
            }
            for g, weight in zip(training, group_weights)
        ],
        "fits": {},
    }
    for kind, matrix in matrices.items():
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            fitted = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=300,
                random_state=SEED,
                fit_intercept=True,
            ).fit(matrix, labels, sample_weight=weights)
        classifiers[kind] = fitted
        artifact = {
            "artifact_version": "paiec-subject-text-offline-nuisance-probe-v1",
            "feature_schema": feature_schema(),
            "vocabulary_": {
                word: int(index) for word, index in vectorizer.vocabulary_.items()
            }
            if kind == "subject_nuisance"
            else {},
            "idf_": vectorizer.idf_.tolist() if kind == "subject_nuisance" else [],
            "subject_coef_": fitted.coef_[0, : text.shape[1]].tolist()
            if kind == "subject_nuisance"
            else [],
            "offline_nuisance_coef_": fitted.coef_[0, -len(names) :].tolist(),
            "intercept_": float(fitted.intercept_[0]),
            "runtime_contrast": [0.0] * len(names),
            "classes_": fitted.classes_.tolist(),
            "prior_strength": 2.0,
        }
        dump(output / f"{kind}_artifact.json", artifact)
        metadata["fits"][kind] = {
            "n_iter": fitted.n_iter_.tolist(),
            "warnings": [str(w.message) for w in observed],
            "artifact_sha256": sha(output / f"{kind}_artifact.json"),
        }
    dump(output / "training.json", metadata)
    return vectorizer, classifiers, text.shape[1]


# 只使用保存的训练参数计算 heldout subject 先验，任务 contrast 始终为零。
def predict_priors(groups, vectorizer, classifiers, names, raw_q):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack

    text = vectorizer.transform([g["document"] for g in groups])
    zeros = csr_matrix(np.zeros((len(groups), len(names))))
    values = {
        "rawq": [raw_q] * len(groups),
        "nuisance": classifiers["nuisance"].predict_proba(zeros)[:, 1].tolist(),
        "subject_nuisance": classifiers["subject_nuisance"]
        .predict_proba(hstack([text, zeros], format="csr"))[:, 1]
        .tolist(),
    }
    return {
        g["group_id"]: {name: p[index] for name, p in values.items()}
        for index, g in enumerate(groups)
    }


# 保存逐组逐预算概率及 N/S 原始统计，三种先验的静态/适应六格用相同评分响应。
def score_fold(groups, priors, output, heldout):
    rows = []
    results = []
    eligible = [g for g in groups if g["eligible"]]
    for cell, (kind, kappa) in CELLS.items():
        alc_values = []
        by_budget = {b: [] for b in BUDGETS}
        for group in eligible:
            prior = priors[group["group_id"]][kind]
            complete = all(group["checkpoints"][str(b)]["complete"] for b in BUDGETS)
            loss = {}
            for budget in BUDGETS:
                checkpoint = group["checkpoints"][str(budget)]
                probability = (
                    prior
                    if kappa is None
                    else (checkpoint["s"] + kappa * prior) / (checkpoint["n"] + kappa)
                )
                value = brier_from_counts(
                    probability, group["evaluation_S"], group["evaluation_N"]
                )
                if checkpoint["complete"]:
                    by_budget[budget].append(value)
                loss[budget] = value
                rows.append(
                    {
                        "heldout": heldout,
                        "cell": cell,
                        "group_id": group["group_id"],
                        "budget": budget,
                        "prior": prior,
                        "prediction": probability,
                        "N": group["evaluation_N"],
                        "S": group["evaluation_S"],
                        "s": checkpoint["s"],
                        "n": checkpoint["n"],
                        "complete": checkpoint["complete"],
                        "brier": value,
                    }
                )
            if complete:
                alc_values.append(
                    sum(w * loss[b] for b, w in zip(BUDGETS, ALC_WEIGHTS))
                )
        result = {
            "heldout": heldout,
            "cell": cell,
            "alc": sum(alc_values) / len(alc_values) if alc_values else None,
            "b0": sum(by_budget[0]) / len(by_budget[0]) if by_budget[0] else None,
            "groups_declared": len(eligible),
            "groups_complete": len(alc_values),
            "missing_budget_entries": sum(
                len(eligible) - len(by_budget[b]) for b in BUDGETS
            ),
            "per_budget": {
                str(b): sum(v) / len(v) if v else None for b, v in by_budget.items()
            },
        }
        results.append(result)
    with (output / "group_budget_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(output / "results.json", results)
    return results


# 在可终止子进程内一次读取全部四源、四折拟合和评分，逐折持久化已完成结果。
def worker(output: Path) -> int:
    started = time.monotonic()
    datasets = {}
    for name, path in SOURCES.items():
        datasets[name] = collect_groups(read_compact(path, name))
    dump(output / "group_statistics.json", datasets)
    all_results = []
    for heldout in SOURCES:
        names = [name for name in SOURCES if name != heldout]
        training = [group for name in names for group in datasets[name]]
        q = sum(g["successes"] for g in training) / sum(
            g["responses"] for g in training
        )
        fold = output / "folds" / heldout
        fold.mkdir(parents=True)
        dump(
            fold / "split.json",
            {
                "heldout": heldout,
                "train_names": names,
                "raw_q": q,
                "training_response_counts": {
                    name: sum(g["responses"] for g in datasets[name]) for name in names
                },
                "target_label_use": "scoring_only; never_fit_or_select",
                "disjoint_benchmarks": heldout not in names,
            },
        )
        vectorizer, classifiers, _ = fit_priors(training, names, fold)
        priors = predict_priors(datasets[heldout], vectorizer, classifiers, names, q)
        dump(fold / "priors.json", priors)
        all_results.extend(score_fold(datasets[heldout], priors, fold, heldout))
        dump(output / "partial_results.json", all_results)
    reference = next(
        r
        for r in all_results
        if r["heldout"] == "truthfulqa_mc" and r["cell"] == "rawq_kappa2"
    )
    old = json.loads(
        (ROOT / "runs/20261014_truthfulqa_validation/summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected = next(
        r["alc"] for r in old["results"] if r["predictor"] == "fixed_kappa2"
    )
    if not math.isclose(reference["alc"], expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError("existing TruthfulQA q kappa2 trajectory mismatch")
    dump(
        output / "worker_result.json",
        {
            "results": all_results,
            "worker_seconds": time.monotonic() - started,
            "truthfulqa_existing_reference": {
                "actual": reference["alc"],
                "expected": expected,
                "absolute_error": abs(reference["alc"] - expected),
            },
        },
    )
    return 0


# 冻结所有实际导入的项目 helper，执行前后均核对字节身份。
def freeze_sources(output: Path) -> dict[str, Any]:
    evidence = {}
    for path in DEPENDENCIES:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        evidence[relative.as_posix()] = {
            "path": str(path),
            "sha256": sha(path),
            "snapshot_sha256": sha(target),
        }
        if (
            evidence[relative.as_posix()]["sha256"]
            != evidence[relative.as_posix()]["snapshot_sha256"]
        ):
            raise ValueError("source snapshot mismatch")
    return evidence


# 总预算包含输入核验、快照、单个可终止worker；未完成格与失败状态明确保留。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cap-seconds", type=int, default=300)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.worker:
        return worker(output)
    if not 0 < args.cap_seconds <= 300 or ROOT not in output.parents:
        parser.error("output must be inside project; cap must be 1..300")
    started = time.monotonic()
    deadline = started + args.cap_seconds
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "planned", "stages": []}
    try:
        inputs = {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
            for name, path in SOURCES.items()
        }
        if any(inputs[name]["sha256"] != EXPECTED[name] for name in SOURCES):
            raise ValueError("input hash mismatch")
        sources = freeze_sources(output)
        import numpy
        import scipy
        import sklearn

        dump(
            output / "plan.json",
            {
                "identity": "regularized-visible-subject-text-offline-centered-nuisance-v1",
                "inputs": inputs,
                "source_snapshot": sources,
                "cap_seconds": args.cap_seconds,
                "seed": SEED,
                "cost_cny": 0,
                "versions": {
                    "python": sys.version,
                    "numpy": numpy.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                },
                "training": {
                    "group_unit": "complete visible subject JSON and benchmark",
                    "response_loss": "all effective recorded responses within group equally weighted; not item-macro",
                    "weights": "Wg=G/(K*Gb); two soft rows weights ybar*Wg,(1-ybar)*Wg; sum=G",
                    "training_short_groups_included": True,
                    "tfidf_documents": "one per training group",
                    "solver": "lbfgs",
                    "C": 1,
                    "max_iter": 300,
                    "fit_intercept": True,
                    "contrast": "e_b-1/K; all zeros at unseen-task inference",
                },
                "controls": list(CELLS),
                "protocol": {
                    "budgets": BUDGETS,
                    "weights": ALC_WEIGHTS,
                    "min_items": MIN_ITEMS,
                    "labeled_visibility": "target_group_only",
                    "response_selection": "first_by_response_id",
                    "compact_input": "full subject; item identity preserved, item content not used; original identity manifest controls source",
                },
                "scope": "one four-fold development probe; no tuning/network/GPU/upload; no ability identifiability claim",
            },
        )
        run_stage(
            output,
            manifest,
            "fit_and_score",
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--output-root",
                str(output),
            ],
            deadline,
        )
        if any(
            sha(Path(value["path"])) != value["sha256"] for value in sources.values()
        ):
            raise ValueError("executed source drift")
        completed = json.loads(
            (output / "worker_result.json").read_text(encoding="utf-8")
        )
        dump(
            output / "summary.json",
            {
                "status": "completed",
                "seconds": time.monotonic() - started,
                "cap_seconds": args.cap_seconds,
                "cost_cny": 0,
                **completed,
                "not_started": [],
            },
        )
        manifest["status"] = "completed"
        dump(output / "manifest.json", manifest)
        return 0
    except Exception as error:
        completed = (
            json.loads((output / "partial_results.json").read_text(encoding="utf-8"))
            if (output / "partial_results.json").exists()
            else []
        )
        done = {(r["heldout"], r["cell"]) for r in completed}
        dump(
            output / "summary.json",
            {
                "status": "timed_out" if isinstance(error, TimeoutError) else "failed",
                "error": type(error).__name__,
                "seconds": time.monotonic() - started,
                "results": completed,
                "not_started": [
                    {"heldout": h, "cell": c}
                    for h in SOURCES
                    for c in CELLS
                    if (h, c) not in done
                ],
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
