"""Reproduce the frozen four-task S4 visible-subject head without evaluation work."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, TOOLS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from feature_ablation.subject_only.features import build_document, feature_schema
from tools.local_evaluation import ResponseRecord, group_records
from tools.run_subject_text_nuisance_study import dump, fit_priors, read_compact, sha

TASKS = ("researchcodebench", "real_webagents", "statqa", "truthfulqa_mc")
EXPECTED_PER_TASK = {
    "researchcodebench": {"groups": 31, "responses": 6784},
    "real_webagents": {"groups": 33, "responses": 4449},
    "statqa": {"groups": 38, "responses": 44194},
    "truthfulqa_mc": {"groups": 150, "responses": 122550},
}
EXPECTED_GROUPS = 252
EXPECTED_RESPONSES = 177977
EXPECTED_ARTIFACT_SHA256 = (
    "bc6961b07044c097c40f6dd2af2c1c4e76a265be16678df563627d0d0f4bfd8e"
)
SCHEMA_VERSION = "paiec-s4-oof-inputs-v1"
RUNTIME_VERSION = "paiec-visible-subject-mixture-v1"
ADAPTATION = {
    "method": "fixed-spike-and-slab",
    "spike_mass": 0.5,
    "slab_strength": 2.0,
}


# 确认新输出目录位于项目内且尚不存在，必须在读取 manifest 或 JSONL 前执行。
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


# 检查一个 manifest 指定路径为项目内绝对常规文件，拒绝相对路径和项目外来源。
def project_input_path(value: object, task: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest path must be a nonempty string for {task}.")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"Manifest path must be absolute for {task}: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Manifest input must be inside project root for {task}: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest input is not a file for {task}: {resolved}")
    return resolved


# 严格解析固定四任务 response_jsonl manifest，并在读取 JSONL 前核验每个字节哈希。
def load_verified_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    resolved_manifest = project_input_path(str(manifest_path.resolve()), "manifest")
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "inputs"}:
        raise ValueError("Manifest must contain exactly schema_version and inputs.")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {manifest['schema_version']!r}")
    entries = manifest["inputs"]
    if not isinstance(entries, list) or len(entries) != len(TASKS):
        raise ValueError("Manifest must contain exactly four ordered S4 inputs.")

    paths: dict[str, Path] = {}
    identities: dict[str, Any] = {}
    for task, entry in zip(TASKS, entries, strict=True):
        if not isinstance(entry, dict) or set(entry) != {
            "task",
            "kind",
            "path",
            "sha256",
        }:
            raise ValueError(
                f"Manifest entry must have exactly four fields for {task}."
            )
        if entry["task"] != task or entry["kind"] != "response_jsonl":
            raise ValueError(
                f"Manifest task or kind differs from fixed S4 order at {task}."
            )
        expected_sha = entry["sha256"]
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"Manifest SHA-256 must have 64 characters for {task}.")
        path = project_input_path(entry["path"], task)
        actual_sha = sha(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Input SHA-256 mismatch for {task}: expected {expected_sha}, got {actual_sha}."
            )
        paths[task] = path
        identities[task] = {
            "path": str(path),
            "kind": "response_jsonl",
            "sha256": actual_sha,
            "bytes": path.stat().st_size,
        }
    return paths, {
        "manifest": {
            "path": str(resolved_manifest),
            "sha256": sha(resolved_manifest),
            "schema_version": SCHEMA_VERSION,
        },
        "tasks": identities,
    }


# 按冻结 deployment training_groups 循环构造全量组，保留短组且不计算任何评测前缀。
def build_training_groups(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: list[dict[str, Any]] = []
    response_counts: dict[str, int] = {}
    for task in TASKS:
        rows: list[ResponseRecord] = read_compact(paths[task], task)
        response_counts[task] = len(rows)
        for key, records in sorted(group_records(rows).items()):
            groups.append(
                {
                    "group_id": hashlib.sha256(
                        f"{key[0]}|{key[1]}".encode()
                    ).hexdigest()[:16],
                    "benchmark": task,
                    "subject": records[0].subject,
                    "document": build_document(records[0].subject, {}),
                    "responses": len(records),
                    "successes": sum(record.label for record in records),
                    "distinct_items": len({record.item_id for record in records}),
                }
            )
    return groups, response_counts


# 断言固定训练样本身份，避免完整四任务拟合被错误的记录数或分组覆盖所替代。
def validate_training_identity(
    groups: list[dict[str, Any]], response_counts: dict[str, int]
) -> None:
    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_GROUPS} training groups, got {len(groups)}."
        )
    if sum(response_counts.values()) != EXPECTED_RESPONSES:
        raise ValueError(
            f"Expected {EXPECTED_RESPONSES} training responses, got {sum(response_counts.values())}."
        )
    for task in TASKS:
        observed_groups = sum(group["benchmark"] == task for group in groups)
        expected = EXPECTED_PER_TASK[task]
        if (
            observed_groups != expected["groups"]
            or response_counts[task] != expected["responses"]
        ):
            raise ValueError(
                f"Unexpected training identity for {task}: "
                f"groups={observed_groups}, responses={response_counts[task]}."
            )


# 将当前 subject_nuisance 内部工件映射为无任务列的冻结运行头，并拒绝字节身份漂移。
def write_runtime_artifact(training_root: Path, output_root: Path) -> dict[str, Any]:
    raw = json.loads(
        (training_root / "subject_nuisance_artifact.json").read_text(encoding="utf-8")
    )
    artifact = {
        "artifact_version": RUNTIME_VERSION,
        "feature_schema": feature_schema(),
        "vocabulary_": raw["vocabulary_"],
        "idf_": raw["idf_"],
        "coef_": raw["subject_coef_"],
        "intercept_": raw["intercept_"],
        "classes_": [0, 1],
        "prior_strength": 2.0,
        "adaptation": ADAPTATION,
    }
    artifact_path = output_root / "artifact.json"
    dump(artifact_path, artifact)
    actual_sha = sha(artifact_path)
    if actual_sha != EXPECTED_ARTIFACT_SHA256:
        raise ValueError(
            "Frozen runtime artifact SHA-256 mismatch: "
            f"expected {EXPECTED_ARTIFACT_SHA256}, got {actual_sha}."
        )
    return {"sha256": actual_sha, "bytes": artifact_path.stat().st_size}


# 保存不含性能指标的训练来源、两次内部拟合和固定组顺序摘要。
def write_fit_summary(
    output_root: Path,
    groups: list[dict[str, Any]],
    response_counts: dict[str, int],
    artifact_identity: dict[str, Any],
) -> None:
    dump(
        output_root / "fit_summary.json",
        {
            "task_order": list(TASKS),
            "training_groups": len(groups),
            "training_responses": sum(response_counts.values()),
            "per_task": {
                task: {
                    "groups": sum(group["benchmark"] == task for group in groups),
                    "responses": response_counts[task],
                }
                for task in TASKS
            },
            "group_order": [
                {"benchmark": group["benchmark"], "group_id": group["group_id"]}
                for group in groups
            ],
            "expected_identity": {
                "groups": EXPECTED_GROUPS,
                "responses": EXPECTED_RESPONSES,
                "per_task": EXPECTED_PER_TASK,
                "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
            },
            "internal_fits": ["nuisance", "subject_nuisance"],
            "runtime_artifact": artifact_identity,
            "performance_evaluation_generated": False,
        },
    )


# 执行完整四任务头的单次拟合；不调用 OOF、校准、预测、评分、封包或 worker。
def reproduce_final_head(input_manifest: Path, output_root: Path) -> dict[str, Any]:
    output_root = validate_new_output_root(output_root)
    paths, input_identity = load_verified_manifest(input_manifest)
    groups, response_counts = build_training_groups(paths)
    validate_training_identity(groups, response_counts)
    for task in TASKS:
        input_identity["tasks"][task]["count"] = response_counts[task]

    output_root.mkdir(parents=True)
    training_root = output_root / "training"
    training_root.mkdir()
    dump(output_root / "input_identity.json", input_identity)
    fit_priors(groups, list(TASKS), training_root)
    artifact_identity = write_runtime_artifact(training_root, output_root)
    write_fit_summary(output_root, groups, response_counts, artifact_identity)
    return {
        "artifact_sha256": artifact_identity["sha256"],
        "training_groups": len(groups),
        "training_responses": sum(response_counts.values()),
    }


# 解析显式 manifest 和新输出目录，不提供默认数据、覆盖或任何评测参数。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Project-internal paiec-s4-oof-inputs-v1 response JSONL manifest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New project-internal output directory; overwrite is unsupported",
    )
    return parser.parse_args(argv)


# 提供仅生成训练工件的 CLI 入口，不启动 worker、预测、评分、校准或封包流程。
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = reproduce_final_head(args.input_manifest, args.output_root)
    print(
        "Wrote final S4 visible-subject head "
        f"({result['training_groups']} groups, {result['training_responses']} responses) "
        f"to {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
