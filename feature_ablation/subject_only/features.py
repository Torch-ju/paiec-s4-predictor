"""Subject-only frozen text template for the feature ablation."""

import json
from typing import Any

FEATURE_SCHEMA_VERSION = "paiec-subject-only-v1"


# 稳定编码完整可见 subject，不读取 item、标识或标签。
def build_document(subject: dict[str, Any], item: dict[str, Any]) -> str:
    return json.dumps(
        subject,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


# 返回与原 artifact 合同兼容的特征 schema。
def feature_schema() -> dict[str, object]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "subject": "complete_visible_subject_dictionary_as_stable_json",
        "item_fields": [],
        "excluded_fields": [
            "benchmark_id",
            "item_id",
            "response_id",
            "label",
            "verifier",
        ],
    }
