"""Frozen, pure-Python text features for the PAIEC cold-start candidate."""

import json
from typing import Any

FEATURE_SCHEMA_VERSION = "paiec-coldstart-text-v1"
FEATURE_TEMPLATE = (
    "subject_config={subject}\n"
    "item_content={item_content}\n"
    "item_features={item_features}\n"
    "interactors={interactors}"
)
ITEM_FEATURE_FIELDS = ("item_content", "item_features", "interactors")


# 将完整可见 subject 配置固定编码为稳定 JSON，避免字典插入顺序改变文本特征。
def _stable_subject(subject: dict[str, Any]) -> str:
    try:
        return json.dumps(
            subject,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "subject must be JSON-serializable without non-finite values."
        ) from exc


# 规范允许的 item 文本字段；不访问 benchmark_id、item_id、response_id 或 label。
def _item_text(item: dict[str, Any], field: str) -> str:
    value = item.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"item field {field!r} must be a string when present.")
    return value


# 按冻结模板生成唯一文档字符串，特征只来自完整 subject 和三项允许的 item 文本。
def build_document(subject: dict[str, Any], item: dict[str, Any]) -> str:
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise TypeError("subject and item must both be dictionaries.")
    return FEATURE_TEMPLATE.format(
        subject=_stable_subject(subject),
        item_content=_item_text(item, "item_content"),
        item_features=_item_text(item, "item_features"),
        interactors=_item_text(item, "interactors"),
    )


# 返回可写入训练 artifact 的冻结特征契约，供推理端拒绝不兼容参数。
def feature_schema() -> dict[str, object]:
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "template": FEATURE_TEMPLATE,
        "subject": "complete_visible_subject_dictionary_as_stable_json",
        "item_fields": list(ITEM_FEATURE_FIELDS),
        "excluded_fields": ["benchmark_id", "item_id", "response_id", "label"],
        "tokenizer": {
            "analyzer": "word",
            "lowercase": True,
            "token_pattern": "(?u)\\b\\w\\w+\\b",
        },
    }
