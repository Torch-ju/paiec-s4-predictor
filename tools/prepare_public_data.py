"""Normalize an organizer-approved PAIEC public-training subset into JSONL.

This tool never contacts Hugging Face. It reads only local Parquet files that
the operator has already obtained under the applicable access conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "3"
SUBJECT_FIELDS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
)
TABLE_SCHEMAS: dict[str, tuple[tuple[str, pa.DataType, bool], ...]] = {
    "subjects": (
        ("subject_id", pa.string(), False),
        ("display_name", pa.string(), False),
        ("normalized_name", pa.string(), True),
        ("provider", pa.string(), True),
        ("release_date", pa.string(), True),
        ("access_date", pa.string(), True),
        ("harness", pa.string(), True),
        ("reasoning_effort", pa.string(), True),
        ("harness_version", pa.string(), True),
        ("subject_features_extra", pa.string(), True),
    ),
    "items": (
        ("item_id", pa.string(), False),
        ("benchmark_id", pa.string(), False),
        ("raw_item_id", pa.string(), False),
        ("content", pa.string(), True),
        ("asset_manifest", pa.string(), True),
        ("grading_criterion", pa.string(), False),
        ("verifier", pa.string(), False),
        ("content_hash", pa.string(), True),
        ("item_features", pa.string(), True),
    ),
    "responses": (
        ("subject_id", pa.string(), False),
        ("item_id", pa.string(), False),
        ("response_id", pa.string(), False),
        ("benchmark_id", pa.string(), False),
        ("trial", pa.int64(), False),
        ("test_condition", pa.string(), True),
        ("interactors", pa.string(), True),
        ("response", pa.float64(), True),
    ),
    "benchmarks": (
        ("benchmark_id", pa.string(), False),
        ("name", pa.string(), False),
        ("version", pa.string(), True),
        ("license", pa.string(), False),
        ("source_url", pa.string(), False),
        ("description", pa.string(), False),
        ("one_line_description", pa.string(), True),
        ("modality", pa.list_(pa.string()), False),
        ("domain", pa.list_(pa.string()), False),
        ("multi_single_turn", pa.string(), False),
        ("response_type", pa.string(), False),
        ("response_scale", pa.string(), False),
        ("categorical", pa.bool_(), False),
        ("paper_url", pa.string(), True),
        ("release_date", pa.string(), True),
        ("granularity", pa.string(), False),
        ("release", pa.string(), True),
        ("n_response_values", pa.int64(), True),
        ("benchmark_features", pa.string(), True),
        ("n_subjects", pa.int64(), False),
        ("n_items", pa.int64(), False),
        ("n_responses", pa.int64(), False),
        ("max_trial", pa.int64(), False),
        ("coverage", pa.float64(), False),
        ("has_reference_answer", pa.bool_(), False),
    ),
}


class PreparationError(RuntimeError):
    """Raised when local input cannot be mapped to the official PAIEC contract."""


# 计算输入或输出文件的 SHA-256，供运行 manifest 追溯实际字节来源。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# 将 schema-v3 可空字符串字段规范为比赛接口要求的普通字符串。
def as_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if not isinstance(value, str):
        raise PreparationError(
            f"expected string field, received {type(value).__name__}"
        )
    return value


# 校验单个 Parquet 的 schema-v3 列名、类型、可空性和版本元数据。
def validate_schema(path: Path, table_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreparationError(f"missing required table: {path}")
    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    version = metadata.get(b"measurement_db.schema_version", b"").decode("utf-8")
    if version != SCHEMA_VERSION:
        raise PreparationError(
            f"{path}: expected measurement_db.schema_version={SCHEMA_VERSION!r}, got {version!r}"
        )
    expected = TABLE_SCHEMAS[table_name]
    actual_names = schema.names
    expected_names = [name for name, _, _ in expected]
    if actual_names != expected_names:
        raise PreparationError(
            f"{path}: schema-v3 columns differ; expected {expected_names}, got {actual_names}"
        )
    fields: list[dict[str, Any]] = []
    for name, expected_type, expected_nullable in expected:
        field = schema.field(name)
        if field.type != expected_type or field.nullable != expected_nullable:
            raise PreparationError(
                f"{path}: field {name!r} must be {expected_type} nullable={expected_nullable}, "
                f"got {field.type} nullable={field.nullable}"
            )
        fields.append(
            {"name": name, "type": str(field.type), "nullable": field.nullable}
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "measurement_db_schema_version": version,
        "fields": fields,
    }


# 读取并验证 allowlist，禁止由目录或公开发布状态自行推断训练资格。
def load_allowlist(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreparationError(f"allowlist does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreparationError(f"allowlist is not valid JSON: {path}") from exc
    if not isinstance(config, dict):
        raise PreparationError("allowlist root must be an object")
    revision = config.get("hf_revision")
    entries = config.get("benchmarks")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision in {"main", "master"}
    ):
        raise PreparationError(
            "allowlist.hf_revision must be a fixed non-branch revision string"
        )
    if not isinstance(entries, list) or not entries:
        raise PreparationError("allowlist.benchmarks must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    directories: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise PreparationError("each allowlist benchmark entry must be an object")
        directory = entry.get("directory")
        source = entry.get("source")
        training_pool = entry.get("training_pool")
        if (
            not isinstance(directory, str)
            or not directory
            or Path(directory).name != directory
            or directory in {".", ".."}
        ):
            raise PreparationError(
                "allowlist directory must be one benchmark directory name"
            )
        if directory in directories:
            raise PreparationError(f"duplicate allowlist directory: {directory}")
        if not isinstance(source, str) or not source.strip():
            raise PreparationError(
                f"{directory}: source must record the authority for training-pool eligibility"
            )
        if not isinstance(training_pool, bool):
            raise PreparationError(f"{directory}: training_pool must be a boolean")
        directories.add(directory)
        normalized.append(
            {"directory": directory, "source": source, "training_pool": training_pool}
        )
    return {
        "hf_revision": revision,
        "benchmarks": normalized,
        "sha256": sha256_file(path),
    }


# 判断 catalog 的二元、逐题且公开发布条件，避免把分数尺度混入二元标签合同。
def catalog_is_binary_item_public(
    catalog: pd.DataFrame, directory: str
) -> tuple[bool, str]:
    if len(catalog) != 1:
        return False, f"{directory}: benchmarks.parquet must contain exactly one row"
    row = catalog.iloc[0]
    if row["release"] != "public":
        return (
            False,
            f"{directory}: catalog release is {row['release']!r}, not 'public'",
        )
    if row["granularity"] != "item":
        return (
            False,
            f"{directory}: catalog granularity is {row['granularity']!r}, not 'item'",
        )
    if row["response_type"] != "binary":
        return (
            False,
            f"{directory}: response_type is {row['response_type']!r}, not 'binary'",
        )
    try:
        scale = json.loads(row["response_scale"])
    except (TypeError, json.JSONDecodeError):
        return False, f"{directory}: response_scale is not valid canonical JSON"
    if scale.get("kind") != "discrete" or scale.get("values") != [0, 1]:
        return False, f"{directory}: binary response_scale must be discrete [0, 1]"
    return True, ""


# 检查各表主键、外键与 benchmark 身份，避免在错误联表上制造训练行。
def validate_relations(
    subjects: pd.DataFrame,
    items: pd.DataFrame,
    responses: pd.DataFrame,
    catalog: pd.DataFrame,
    directory: str,
) -> str:
    benchmark_id = as_string(catalog.iloc[0]["benchmark_id"])
    for name, table, key in (
        ("subjects", subjects, "subject_id"),
        ("items", items, "item_id"),
        ("responses", responses, "response_id"),
    ):
        if table[key].isna().any() or table[key].duplicated().any():
            raise PreparationError(
                f"{directory}: {name}.{key} must be non-null and unique"
            )
    if not (items["benchmark_id"] == benchmark_id).all():
        raise PreparationError(f"{directory}: item benchmark_id does not match catalog")
    if not (responses["benchmark_id"] == benchmark_id).all():
        raise PreparationError(
            f"{directory}: response benchmark_id does not match catalog"
        )
    if not responses["subject_id"].isin(subjects["subject_id"]).all():
        raise PreparationError(
            f"{directory}: responses reference an unknown subject_id"
        )
    if not responses["item_id"].isin(items["item_id"]).all():
        raise PreparationError(f"{directory}: responses reference an unknown item_id")
    return benchmark_id


# 将一个已确认基准的逐次响应转为比赛接口所需的 JSONL 记录并统计显式过滤原因。
def normalize_benchmark(
    directory: str,
    subjects: pd.DataFrame,
    items: pd.DataFrame,
    responses: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    benchmark_id = validate_relations(subjects, items, responses, catalog, directory)
    subject_by_id = subjects.set_index("subject_id", verify_integrity=True)
    item_by_id = items.set_index("item_id", verify_integrity=True)
    records: list[dict[str, Any]] = []
    counts = {
        "responses_seen": len(responses),
        "missing_label": 0,
        "non_binary_label": 0,
    }
    for response in responses.itertuples(index=False):
        value = response.response
        if value is None or pd.isna(value):
            counts["missing_label"] += 1
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
            counts["non_binary_label"] += 1
            continue
        subject = subject_by_id.loc[response.subject_id]
        item = item_by_id.loc[response.item_id]
        records.append(
            {
                "subject": {
                    field: as_string(subject[field]) for field in SUBJECT_FIELDS
                },
                "item": {
                    "item_content": as_string(item["content"]),
                    "item_features": as_string(item["item_features"]),
                    "interactors": as_string(response.interactors),
                    "benchmark_id": benchmark_id,
                },
                "label": int(numeric),
                "response_id": as_string(response.response_id),
                "item_id": as_string(response.item_id),
                "benchmark_id": benchmark_id,
            }
        )
    counts["records_exported"] = len(records)
    return records, counts, benchmark_id


# 原子写入 JSON 或 JSONL，防止校验失败或中断留下半截交付物。
def atomic_write(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise PreparationError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


# 执行 allowlist 驱动的本地联表、过滤、JSONL 导出和可追溯 manifest 生成。
def prepare_public_data(
    data_root: Path,
    allowlist_path: Path,
    output_path: Path,
    manifest_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_path.resolve() == manifest_path.resolve():
        raise PreparationError("output and manifest paths must differ")
    allowlist = load_allowlist(allowlist_path)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    all_records: list[dict[str, Any]] = []
    total_counts = {
        "responses_seen": 0,
        "missing_label": 0,
        "non_binary_label": 0,
        "records_exported": 0,
    }

    for entry in allowlist["benchmarks"]:
        directory = entry["directory"]
        if not entry["training_pool"]:
            skipped.append(
                {"directory": directory, "reason": "training_pool_not_confirmed"}
            )
            continue
        tables = {
            name: data_root / directory / "formatted_tables" / f"{name}.parquet"
            for name in TABLE_SCHEMAS
        }
        table_manifest = {
            name: validate_schema(path, name) for name, path in tables.items()
        }
        subjects = pd.read_parquet(tables["subjects"])
        items = pd.read_parquet(tables["items"])
        responses = pd.read_parquet(tables["responses"])
        catalog = pd.read_parquet(tables["benchmarks"])
        eligible, reason = catalog_is_binary_item_public(catalog, directory)
        if not eligible:
            skipped.append({"directory": directory, "reason": reason})
            continue
        records, counts, benchmark_id = normalize_benchmark(
            directory, subjects, items, responses, catalog
        )
        all_records.extend(records)
        for key, value in counts.items():
            total_counts[key] += value
        selected.append(
            {
                "directory": directory,
                "benchmark_id": benchmark_id,
                "source": entry["source"],
                "training_pool": True,
                "catalog_release": "public",
                "response_type": "binary",
                "granularity": "item",
                "tables": table_manifest,
                "counts": counts,
            }
        )

    if not selected:
        raise PreparationError(
            "no allowlisted benchmark met training_pool + public binary item-level conditions"
        )

    jsonl = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
        for record in all_records
    )
    atomic_write(output_path, jsonl, overwrite)
    output_hash = sha256_file(output_path)
    manifest = {
        "purpose": "paiec_public_training_jsonl_normalization",
        "input_root": str(data_root),
        "allowlist": {
            "path": str(allowlist_path),
            "sha256": allowlist["sha256"],
            "hf_revision": allowlist["hf_revision"],
        },
        "schema": {
            "measurement_db_schema_version": SCHEMA_VERSION,
            "tables": list(TABLE_SCHEMAS),
        },
        "selected_benchmarks": selected,
        "skipped_benchmarks": skipped,
        "totals": total_counts,
        "output": {
            "path": str(output_path),
            "sha256": output_hash,
            "records": len(all_records),
        },
        "limitations": [
            "The hf_revision is an operator-confirmed allowlist value; local Parquet alone cannot prove its repository revision.",
            "A catalog release value of public does not establish PAIEC training-pool eligibility without training_pool=true and its recorded authority.",
            "This preparation validates local schema-v3 files only; it does not download data, access Codabench, or measure model performance.",
        ],
    }
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        overwrite,
    )
    return manifest


# 提供受限本地数据准备的命令行入口，不触发网络、登录或远端下载。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="local root containing <directory>/formatted_tables/",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        required=True,
        help="operator-confirmed JSON allowlist with a fixed hf_revision",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="manifest JSON output path"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output and manifest"
    )
    args = parser.parse_args()
    manifest = prepare_public_data(
        data_root=args.data_root,
        allowlist_path=args.allowlist,
        output_path=args.output,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "records": manifest["output"]["records"],
                "output": manifest["output"]["path"],
            }
        )
    )


if __name__ == "__main__":
    main()
