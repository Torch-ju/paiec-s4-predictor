"""Fit four S4 leave-one-task OOF subject priors and one shared provider calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_ablation.subject_only.features import build_document
from tools.run_subject_text_nuisance_study import (
    collect_groups,
    fit_priors,
    predict_priors,
    read_compact,
    sha,
)

TASKS = ("researchcodebench", "real_webagents", "statqa", "truthfulqa_mc")
SUBJECT_FIELDS = {
    "access_date",
    "harness",
    "harness_version",
    "normalized_name",
    "provider",
    "reasoning_effort",
    "release_date",
    "subject_features_extra",
}
CAP_SECONDS = 20
L2 = 1.0
CLOSURE_SOURCES = (
    Path(__file__).resolve(),
    ROOT / "tools/run_subject_text_nuisance_study.py",
    ROOT / "tools/local_evaluation.py",
    ROOT / "tools/run_kappa_selection.py",
    ROOT / "tools/run_multibench_study.py",
    ROOT / "tools/train_coldstart.py",
    ROOT / "coldstart/features.py",
    ROOT / "feature_ablation/subject_only/features.py",
)


# 原子保存结构化回执，避免半写 JSON 被后续审计误读为完成证据。
def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


# 验证一个有限数值，区分布尔值和真正的计数或概率参数。
def finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


# 校验完整官方 subject 八字段及稳定 document，拒绝任何静默字段补全或标识替代。
def validate_group(group: object, task: str, all_ids: set[str]) -> dict[str, Any]:
    if not isinstance(group, dict):
        raise TypeError(f"{task} group must be an object")
    required = {
        "group_id",
        "subject",
        "document",
        "responses",
        "successes",
        "distinct_items",
        "eligible",
    }
    if set(group) != required:
        raise ValueError(f"{task} group schema differs from the fixed contract")
    group_id = group["group_id"]
    subject = group["subject"]
    if not isinstance(group_id, str) or not group_id or group_id in all_ids:
        raise ValueError("group_id must be a unique nonempty string across four tasks")
    if not isinstance(subject, dict) or set(subject) != SUBJECT_FIELDS:
        raise ValueError("subject must contain exactly the eight approved fields")
    if any(not isinstance(value, str) for value in subject.values()):
        raise TypeError("all subject fields must be strings")
    if group["document"] != build_document(subject, {}):
        raise ValueError("group document differs from authoritative build_document")
    values = {
        name: group[name] for name in ("responses", "successes", "distinct_items")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise TypeError("group counts must be non-boolean integers")
    responses, successes, distinct_items = (
        values["responses"],
        values["successes"],
        values["distinct_items"],
    )
    if (
        not 0 < responses
        or not 0 <= successes <= responses
        or not 0 < distinct_items <= responses
    ):
        raise ValueError("group counts are inconsistent")
    eligible = distinct_items >= 80
    if type(group["eligible"]) is not bool or group["eligible"] is not eligible:
        raise ValueError("eligible must exactly equal distinct_items >= 80")
    all_ids.add(group_id)
    return {
        "group_id": group_id,
        "benchmark": task,
        "subject": subject,
        "document": group["document"],
        "responses": responses,
        "successes": successes,
        "distinct_items": distinct_items,
        "eligible": eligible,
    }


# 读取 group-JSON 输入并进行完整 schema、跨任务 group_id 与资格口径核验。
def load_group_json(path: Path, task: str, all_ids: set[str]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid group_json for {task}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "task",
        "groups",
    }:
        raise ValueError("group_json schema differs from the fixed contract")
    if payload["schema_version"] != "paiec-s4-group-json-v1" or payload["task"] != task:
        raise ValueError("group_json task identity differs from manifest")
    groups = payload["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("group_json must contain nonempty groups")
    return [validate_group(group, task, all_ids) for group in groups]


# 读取响应 JSONL 后复用权威紧凑读取与聚合，不执行旧研究 worker 或评分入口。
def load_response_jsonl(
    path: Path, task: str, all_ids: set[str]
) -> list[dict[str, Any]]:
    groups = collect_groups(read_compact(path, task))
    normalized = []
    for group in groups:
        payload = {
            "group_id": group["group_id"],
            "subject": group["subject"],
            "document": group["document"],
            "responses": group["responses"],
            "successes": group["successes"],
            "distinct_items": group["distinct_items"],
            "eligible": group["eligible"],
        }
        normalized.append(validate_group(payload, task, all_ids))
    return normalized


# 解析固定四任务 manifest，并在读入数据前以 SHA-256 绑定每个显式输入文件。
def load_manifest(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("input manifest is not valid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "inputs"}
        or manifest["schema_version"] != "paiec-s4-oof-inputs-v1"
    ):
        raise ValueError("input manifest schema differs from the fixed contract")
    entries = manifest["inputs"]
    if not isinstance(entries, list) or len(entries) != len(TASKS):
        raise ValueError("input manifest must bind exactly four tasks")
    if [
        entry.get("task") if isinstance(entry, dict) else None for entry in entries
    ] != list(TASKS):
        raise ValueError("input manifest task order must be the fixed S4 order")
    datasets, inputs, all_ids = {}, {}, set()
    for entry, task in zip(entries, TASKS):
        if not isinstance(entry, dict) or set(entry) != {
            "task",
            "path",
            "sha256",
            "kind",
        }:
            raise ValueError("each manifest input must contain task/path/sha256/kind")
        if entry["kind"] not in {"group_json", "response_jsonl"}:
            raise ValueError("manifest kind must be group_json or response_jsonl")
        if not isinstance(entry["path"], str) or not isinstance(entry["sha256"], str):
            raise TypeError("manifest path and sha256 must be strings")
        source = (path.parent / entry["path"]).resolve()
        if source != ROOT and ROOT not in source.parents:
            raise ValueError("manifest input path must remain inside the project")
        if not source.is_file() or sha(source) != entry["sha256"].lower():
            raise ValueError(f"input SHA-256 mismatch for {task}")
        groups = (
            load_group_json(source, task, all_ids)
            if entry["kind"] == "group_json"
            else load_response_jsonl(source, task, all_ids)
        )
        if not any(group["eligible"] for group in groups):
            raise ValueError(f"{task} has no eligible groups")
        datasets[task] = groups
        inputs[task] = {
            "path": str(source),
            "sha256": sha(source),
            "bytes": source.stat().st_size,
            "kind": entry["kind"],
            "groups": len(groups),
            "eligible_groups": sum(group["eligible"] for group in groups),
        }
    return datasets, {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha(path),
        "tasks": inputs,
    }


# 用原 pilot 的固定 lambda=1 Bernoulli NLL 计算共享 logit 偏移及其一阶、二阶导数。
def objective_gradient_hessian(
    theta: float, rows: list[dict[str, Any]]
) -> tuple[float, float, float]:
    objective, gradient, hessian = L2 * theta * theta / 2.0, L2 * theta, L2
    for row in rows:
        prior = finite_real(row["prior"], "OOF prior")
        if not 0.0 < prior < 1.0:
            raise ValueError("OOF prior must lie strictly inside (0, 1)")
        score = math.log(prior / (1.0 - prior)) + theta
        probability = (
            1.0 / (1.0 + math.exp(-score))
            if score >= 0
            else math.exp(score) / (1.0 + math.exp(score))
        )
        rate = row["successes"] / row["responses"]
        weight = finite_real(row["weight"], "calibration weight")
        objective += weight * (math.log1p(math.exp(score)) - rate * score)
        gradient += weight * (probability - rate)
        hessian += weight * probability * (1.0 - probability)
    return objective, gradient, hessian


# 对非空 provider 的不重归一固定权重执行 brentq[-1,1]，语义与原 pilot 一维共享控制相同。
def fit_shared_offset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.optimize import brentq

    visible = [row for row in rows if row["subject"]["provider"]]
    if not visible:
        raise ValueError("shared calibration has no nonempty-provider rows")
    left_gradient = objective_gradient_hessian(-1.0, visible)[1]
    right_gradient = objective_gradient_hessian(1.0, visible)[1]
    if left_gradient > 0.0 or right_gradient < 0.0:
        raise ValueError("shared calibration gradient does not bracket zero on [-1, 1]")
    theta, result = brentq(
        lambda value: objective_gradient_hessian(value, visible)[1],
        -1.0,
        1.0,
        xtol=1e-14,
        rtol=1e-14,
        maxiter=100,
        full_output=True,
        disp=False,
    )
    objective, gradient, hessian = objective_gradient_hessian(theta, visible)
    if not result.converged or abs(gradient) > 1e-10:
        raise RuntimeError("shared calibration brentq did not converge")
    return {
        "theta": theta,
        "objective": objective,
        "gradient": gradient,
        "hessian": hessian,
        "iterations": result.iterations,
        "groups": len(visible),
        "tasks": list(TASKS),
        "seen_providers": sorted({row["subject"]["provider"] for row in visible}),
        "solver": {
            "name": "scipy.optimize.brentq",
            "bracket": [-1.0, 1.0],
            "xtol": 1e-14,
            "rtol": 1e-14,
            "maxiter": 100,
            "left_gradient": left_gradient,
            "right_gradient": right_gradient,
        },
    }


# 在四个留一任务方向复用原训练器与零 contrast 预测器，输出仅 eligible group_id 到标量先验映射。
def run_worker(manifest_path: Path, output: Path, deadline: float) -> None:
    datasets, input_identity = load_manifest(manifest_path)
    folds, calibration_rows = {}, []
    for heldout in TASKS:
        training_names = [task for task in TASKS if task != heldout]
        training = [group for task in training_names for group in datasets[task]]
        fold = output / "folds" / heldout
        training_dir = fold / "training"
        training_dir.mkdir(parents=True)
        vectorizer, classifiers, _ = fit_priors(training, training_names, training_dir)
        raw_q = sum(group["successes"] for group in training) / sum(
            group["responses"] for group in training
        )
        eligible = [group for group in datasets[heldout] if group["eligible"]]
        priors = predict_priors(
            eligible, vectorizer, classifiers, training_names, raw_q
        )
        scalar_map = {
            group_id: float(values["subject_nuisance"])
            for group_id, values in priors.items()
        }
        if set(scalar_map) != {group["group_id"] for group in eligible}:
            raise ValueError(
                "eligible-only OOF map does not match heldout eligible groups"
            )
        if any(
            not 0.0 < prior < 1.0 or not math.isfinite(prior)
            for prior in scalar_map.values()
        ):
            raise ValueError(
                "subject-nuisance OOF priors must be finite interior probabilities"
            )
        dump(fold / "oof_subject_nuisance.json", scalar_map)
        weights = [
            len(training)
            / (
                len(training_names)
                * sum(group["benchmark"] == task for group in training)
            )
            for task in training_names
        ]
        folds[heldout] = {
            "heldout": heldout,
            "training_names": training_names,
            "training_groups": len(training),
            "training_includes_short_groups": any(
                not group["eligible"] for group in training
            ),
            "per_task_group_weights": dict(zip(training_names, weights)),
            "eligible_heldout_groups": len(eligible),
            "oof_map_sha256": sha(fold / "oof_subject_nuisance.json"),
            "nuisance_artifact_sha256": sha(training_dir / "nuisance_artifact.json"),
            "subject_nuisance_artifact_sha256": sha(
                training_dir / "subject_nuisance_artifact.json"
            ),
            "nuisance_recorded_not_scored": True,
        }
        weight = 1.0 / (len(TASKS) * len(eligible))
        calibration_rows.extend(
            {
                "heldout": heldout,
                "group_id": group["group_id"],
                "subject": group["subject"],
                "successes": group["successes"],
                "responses": group["responses"],
                "prior": scalar_map[group["group_id"]],
                "weight": weight,
            }
            for group in eligible
        )
        if time.monotonic() > deadline:
            raise TimeoutError("S4 OOF worker exceeded cap")
    if not math.isclose(
        sum(row["weight"] for row in calibration_rows), 1.0, abs_tol=1e-12
    ):
        raise ValueError("four-task calibration weights do not sum to one")
    calibration = fit_shared_offset(calibration_rows)
    import numpy
    import scipy
    import sklearn

    dump(
        output / "summary.json",
        {
            "status": "completed",
            "scope": "S4 training-side OOF and shared calibration only; no scoring, target run, pi learning, or package output",
            "input_identity": input_identity,
            "task_order": list(TASKS),
            "folds": folds,
            "calibration_fit": calibration,
            "fixed_calibration_weight_total": sum(
                row["weight"] for row in calibration_rows
            ),
            "nonempty_provider_weight_total": sum(
                row["weight"] for row in calibration_rows if row["subject"]["provider"]
            ),
            "software": {
                "python": sys.version,
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
            "nuisance_artifacts_recorded_not_scored": True,
        },
    )
    dump(
        output / "calibration.json",
        {
            "calibration_version": "paiec-provider-agnostic-masked-shared-v1",
            "method": "provider-agnostic-shared-logit-offset",
            "theta": calibration["theta"],
            "seen_providers": calibration["seen_providers"],
            "source_summary_sha256": sha(output / "summary.json"),
            "source_input_manifest_sha256": input_identity["manifest_sha256"],
            "cmmlu_labels_in_shared_theta": False,
        },
    )


# 冻结本入口及 fit_priors 真实本地传递闭包的八个源码，记录副本哈希而不复制训练输入。
def freeze_sources(output: Path) -> dict[str, dict[str, str]]:
    evidence = {}
    for source in CLOSURE_SOURCES:
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        evidence[relative.as_posix()] = {
            "sha256": sha(source),
            "snapshot_sha256": sha(target),
        }
        if (
            evidence[relative.as_posix()]["sha256"]
            != evidence[relative.as_posix()]["snapshot_sha256"]
        ):
            raise ValueError("source snapshot mismatch")
    return evidence


# 核验父进程建立的八源码快照和输入 manifest，worker 开始前还拒绝任何已有结果工件。
def verify_prepared_identity(output: Path, require_fresh: bool) -> dict[str, Any]:
    required = {"input_config.json", "source_identity.json", "plan.json", "source"}
    result_names = {
        "folds",
        "summary.json",
        "calibration.json",
        "failure.json",
        "execution_command.json",
        "worker.stdout.log",
        "worker.stderr.log",
    }
    if not output.is_dir():
        raise FileNotFoundError("worker output-root was not created by the parent")
    present = {path.name for path in output.iterdir()}
    if require_fresh and (present != required or present & result_names):
        raise ValueError("worker output-root is not a fresh parent-prepared directory")
    if not required <= present:
        raise ValueError("worker output-root lacks parent-prepared identity files")
    input_identity = json.loads(
        (output / "input_config.json").read_text(encoding="utf-8")
    )
    source_identity = json.loads(
        (output / "source_identity.json").read_text(encoding="utf-8")
    )
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if source_identity.get("runner_sha256") != sha(Path(__file__).resolve()):
        raise ValueError("runner source drifted after parent preparation")
    sources = source_identity.get("sources")
    expected = {source.relative_to(ROOT).as_posix() for source in CLOSURE_SOURCES}
    if not isinstance(sources, dict) or set(sources) != expected:
        raise ValueError(
            "source closure identity does not contain the required eight files"
        )
    for relative, identity in sources.items():
        live, snapshot = ROOT / relative, output / "source" / relative
        if (
            not isinstance(identity, dict)
            or not live.is_file()
            or not snapshot.is_file()
            or sha(live) != identity.get("sha256")
            or sha(snapshot) != identity.get("snapshot_sha256")
            or identity["sha256"] != identity["snapshot_sha256"]
        ):
            raise ValueError(f"source identity drift: {relative}")
    manifest_path = Path(input_identity.get("manifest_path", ""))
    if not manifest_path.is_file() or sha(manifest_path) != input_identity.get(
        "manifest_sha256"
    ):
        raise ValueError("input manifest identity drift")
    tasks = input_identity.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise ValueError("input task identity differs from fixed S4 tasks")
    for task, identity in tasks.items():
        source = Path(identity.get("path", "")) if isinstance(identity, dict) else None
        if not source or not source.is_file() or sha(source) != identity.get("sha256"):
            raise ValueError(f"input source identity drift: {task}")
    if (
        plan.get("status") != "planned"
        or plan.get("task_order") != list(TASKS)
        or plan.get("input_manifest_sha256") != input_identity["manifest_sha256"]
    ):
        raise ValueError("parent plan identity differs from input configuration")
    return {
        "source_files": len(sources),
        "input_manifest_sha256": input_identity["manifest_sha256"],
    }


# 主进程先拒绝旧输出，再实际监督独立 worker 子进程的二十秒总上限并保存 stdout/stderr。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=int, default=CAP_SECONDS)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not 0 < args.cap_seconds <= CAP_SECONDS or ROOT not in output.parents:
        parser.error("cap-seconds must be 1..20 and output-root must be inside project")
    if args.worker:
        verify_prepared_identity(output, require_fresh=True)
        prepared_input = json.loads(
            (output / "input_config.json").read_text(encoding="utf-8")
        )
        if (
            args.input_manifest.resolve()
            != Path(prepared_input["manifest_path"]).resolve()
        ):
            raise ValueError(
                "worker input manifest differs from parent-prepared identity"
            )
        run_worker(
            args.input_manifest.resolve(), output, time.monotonic() + args.cap_seconds
        )
        return 0
    if output.exists():
        raise FileExistsError(
            "output-root already exists; refusing before input read or fitting"
        )
    started = time.monotonic()
    output.mkdir(parents=True)
    process = None
    stdout, stderr = "", ""
    try:
        datasets, input_identity = load_manifest(args.input_manifest.resolve())
        sources = freeze_sources(output)
        dump(output / "input_config.json", input_identity)
        dump(
            output / "source_identity.json",
            {"sources": sources, "runner_sha256": sha(Path(__file__).resolve())},
        )
        dump(
            output / "plan.json",
            {
                "status": "planned",
                "cap_seconds": args.cap_seconds,
                "cost_cny": 0,
                "task_order": list(TASKS),
                "input_manifest_sha256": input_identity["manifest_sha256"],
                "input_tasks": {
                    task: {
                        "groups": len(groups),
                        "eligible": sum(group["eligible"] for group in groups),
                    }
                    for task, groups in datasets.items()
                },
                "scope": "four S4 leave-one-task OOF fits plus one fixed shared calibration; no scoring, pi learning, target run, network, or package output",
            },
        )
        pre_worker_identity = verify_prepared_identity(output, require_fresh=True)
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--input-manifest",
            str(args.input_manifest.resolve()),
            "--output-root",
            str(output),
            "--cap-seconds",
            str(args.cap_seconds),
        ]
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            stdout, stderr = process.communicate(
                timeout=max(0.1, args.cap_seconds - (time.monotonic() - started))
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            (output / "worker.stdout.log").write_text(stdout, encoding="utf-8")
            (output / "worker.stderr.log").write_text(stderr, encoding="utf-8")
            raise TimeoutError(
                "S4 OOF worker exceeded parent supervision cap"
            ) from None
        (output / "worker.stdout.log").write_text(stdout, encoding="utf-8")
        (output / "worker.stderr.log").write_text(stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"S4 OOF worker exited {process.returncode}")
        post_worker_identity = verify_prepared_identity(output, require_fresh=False)
        dump(
            output / "execution_command.json",
            {
                "command": command,
                "pid": process.pid,
                "exit_code": process.returncode,
                "seconds": time.monotonic() - started,
                "cap_seconds": args.cap_seconds,
                "cost_cny": 0,
                "worker_source_sha256": sha(Path(__file__).resolve()),
                "pre_worker_identity": pre_worker_identity,
                "post_worker_identity": post_worker_identity,
            },
        )
        return 0
    except Exception as error:
        dump(
            output / "failure.json",
            {
                "status": "timed_out" if isinstance(error, TimeoutError) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.monotonic() - started,
                "cap_seconds": args.cap_seconds,
                "cost_cny": 0,
                "worker": {
                    "pid": process.pid if process is not None else None,
                    "returncode": process.returncode if process is not None else None,
                    "source_sha256": sha(Path(__file__).resolve()),
                },
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
