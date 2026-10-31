"""在公开 JSONL 记录上复现 PAIEC 的分组少标签概率评测核心。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


@dataclass(frozen=True)
class ResponseRecord:
    """保存一条已规范化的公开响应，并保留其分组与追溯标识。"""

    subject: dict[str, Any]
    item: dict[str, Any]
    label: int
    response_id: str
    item_id: str
    benchmark_id: str


@dataclass
class GroupResult:
    """汇总一个 subject–benchmark 组合在六档预算下的局部结果。"""

    group_id: str
    benchmark_id: str
    distinct_items: int
    acquisition_items: int
    evaluation_records: int
    budgets: dict[int, dict[str, Any]]


# 将可嵌套 JSON 值规范成稳定字符串，供分组、缓存和哈希复用。
def stable_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "subject、item 与配置必须是可序列化且不含非有限数的 JSON 值"
        ) from exc


# 计算文件字节哈希，记录实际数据和模型来源而不读取其语义内容。
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 验证并转换模型输出，拒绝非有限值和越界的伪概率。
def validate_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("predict 必须返回数值概率，不能返回布尔值")
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("predict 必须返回可转换为浮点数的概率") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("predict 必须返回有限且位于 [0, 1] 的概率")
    return probability


# 从独立 model.py 装载 predict 接口，不依赖项目级包安装。
def load_predictor(path: Path) -> Callable[[list[dict[str, Any]], list[Any]], float]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{path}")
    module_name = f"paiec_local_model_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法装载模型文件：{path}")
    module = importlib.util.module_from_spec(spec)
    previous_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise TypeError(f"{path} 必须定义可调用的 predict(input, labeled)")
    return predict


# 规范单条 JSONL 输入，并确保评测键在记录和 item 字典中一致。
def normalize_record(raw: Any, line_number: int) -> ResponseRecord:
    if not isinstance(raw, dict):
        raise TypeError(f"第 {line_number} 行必须是 JSON 对象")
    required = ("subject", "item", "label", "response_id", "item_id", "benchmark_id")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"第 {line_number} 行缺少字段：{', '.join(missing)}")
    subject, item = raw["subject"], raw["item"]
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise TypeError(f"第 {line_number} 行的 subject 与 item 必须是对象")
    if type(raw["label"]) is not int or raw["label"] not in (0, 1):
        raise ValueError(f"第 {line_number} 行的 label 必须是二元整数 0 或 1")
    response_id, item_id, benchmark_id = (
        raw[name] for name in ("response_id", "item_id", "benchmark_id")
    )
    if not all(
        isinstance(value, str) and value
        for value in (response_id, item_id, benchmark_id)
    ):
        raise ValueError(
            f"第 {line_number} 行的 response_id、item_id、benchmark_id 必须为非空字符串"
        )
    item_copy = copy.deepcopy(item)
    embedded_benchmark = item_copy.get("benchmark_id")
    if embedded_benchmark is not None and embedded_benchmark != benchmark_id:
        raise ValueError(
            f"第 {line_number} 行的 item.benchmark_id 与 benchmark_id 不一致"
        )
    item_copy["benchmark_id"] = benchmark_id
    stable_json(subject)
    stable_json(item_copy)
    return ResponseRecord(
        copy.deepcopy(subject),
        item_copy,
        raw["label"],
        response_id,
        item_id,
        benchmark_id,
    )


# 读取公开训练池导出的 JSONL，不接受空文件或重复 response_id。
def load_records(path: Path) -> list[ResponseRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到 JSONL 数据：{path}")
    records: list[ResponseRecord] = []
    response_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效 JSON") from exc
            record = normalize_record(raw, line_number)
            if record.response_id in response_ids:
                raise ValueError(f"response_id 重复：{record.response_id}")
            response_ids.add(record.response_id)
            records.append(record)
    if not records:
        raise ValueError("JSONL 中没有可评测记录")
    return records


# 以完整 subject 稳定 JSON 和 benchmark_id 分组，避免仅按模型名造成混组。
def group_records(
    records: Iterable[ResponseRecord],
) -> dict[tuple[str, str], list[ResponseRecord]]:
    groups: dict[tuple[str, str], list[ResponseRecord]] = defaultdict(list)
    for record in records:
        groups[(stable_json(record.subject), record.benchmark_id)].append(record)
    return dict(groups)


# 从种子和组键导出局部随机数种子，使输入行顺序不改变拆分结果。
def group_seed(seed: int, key: tuple[str, str]) -> int:
    payload = f"{seed}|{key[0]}|{key[1]}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# 以 item 为原子做固定 50/50 划分，保证同 item 的所有响应永不跨侧。
def split_items(
    records: list[ResponseRecord], seed: int, key: tuple[str, str]
) -> tuple[list[str], list[str], dict[str, list[ResponseRecord]]]:
    by_item: dict[str, list[ResponseRecord]] = defaultdict(list)
    item_inputs: dict[str, str] = {}
    for record in records:
        item_input = stable_json(record.item)
        previous = item_inputs.get(record.item_id)
        if previous is not None and previous != item_input:
            raise ValueError(
                f"同组 item_id {record.item_id!r} 对应不一致的完整 item 输入，无法安全复用预测或划分"
            )
        item_inputs[record.item_id] = item_input
        by_item[record.item_id].append(record)
    item_ids = sorted(by_item)
    if len(item_ids) < 2:
        raise ValueError("每个参与评测的组至少需要两个 distinct items")
    rng = random.Random(group_seed(seed, key))
    rng.shuffle(item_ids)
    acquisition_size = len(item_ids) // 2
    acquisition_ids = item_ids[:acquisition_size]
    evaluation_ids = item_ids[acquisition_size:]
    return acquisition_ids, evaluation_ids, dict(by_item)


# 生成模型可见的受限标签，始终复制数据以防外部模型修改评测状态。
def labeled_payload(records: list[ResponseRecord]) -> list[list[Any]]:
    return [
        [[copy.deepcopy(record.subject), copy.deepcopy(record.item)], record.label]
        for record in records
    ]


# 为相同 subject–item 输入复用一次预测，记录调用与复用次数。
def score_evaluation_records(
    predictor: Callable[[list[dict[str, Any]], list[Any]], float],
    records: list[ResponseRecord],
    labels: list[ResponseRecord],
    budget: int,
    checkpoint_complete: bool,
) -> tuple[list[dict[str, Any]], dict[str, float | int | None]]:
    cache: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    calls = 0
    reusable_labels = labeled_payload(labels)
    for record in records:
        model_input = [copy.deepcopy(record.subject), copy.deepcopy(record.item)]
        cache_key = stable_json(model_input)
        if cache_key not in cache:
            cache[cache_key] = validate_probability(
                predictor(model_input, copy.deepcopy(reusable_labels))
            )
            calls += 1
        prediction = cache[cache_key]
        brier = (prediction - record.label) ** 2
        rows.append(
            {
                "budget": budget,
                "checkpoint_complete": checkpoint_complete,
                "labels_available": len(labels),
                "response_id": record.response_id,
                "item_id": record.item_id,
                "benchmark_id": record.benchmark_id,
                "label": record.label,
                "prediction": prediction,
                "brier": brier,
                "ece_bin": min(9, int(prediction * 10)),
            }
        )
    if not rows:
        raise ValueError("evaluation 侧没有响应，无法计算分数")
    brier_score = sum(float(row["brier"]) for row in rows) / len(rows)
    ece = expected_calibration_error(rows)
    return rows, {
        "brier": brier_score,
        "ece": ece,
        "prediction_calls": calls,
        "prediction_reuses": len(rows) - calls,
    }


# 按十个固定概率桶计算 ECE，仅作校准诊断而不替代 Brier ALC。
def expected_calibration_error(rows: list[dict[str, Any]]) -> float:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for row in rows:
        buckets[int(row["ece_bin"])].append(row)
    total = len(rows)
    return sum(
        (len(bucket) / total)
        * abs(
            sum(float(row["prediction"]) for row in bucket) / len(bucket)
            - sum(int(row["label"]) for row in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


# 为一个组构造嵌套预算轨迹，默认策略随机但无需未来标签。
def evaluate_group(
    key: tuple[str, str],
    records: list[ResponseRecord],
    predictor: Callable[[list[dict[str, Any]], list[Any]], float],
    seed: int,
    max_labels: int = 31,
) -> tuple[GroupResult, list[dict[str, Any]]]:
    if max_labels != max(BUDGETS):
        raise ValueError(
            f"本地核心当前要求 max_labels={max(BUDGETS)}，以匹配六档官方预算"
        )
    acquisition_ids, evaluation_ids, by_item = split_items(records, seed, key)
    evaluation_records = [
        record for item_id in evaluation_ids for record in by_item[item_id]
    ]
    if not evaluation_records:
        raise ValueError("item 划分后 evaluation 侧为空")
    rng = random.Random(group_seed(seed + 1, key))
    candidate_ids = list(acquisition_ids)
    rng.shuffle(candidate_ids)
    acquired: list[ResponseRecord] = []
    all_rows: list[dict[str, Any]] = []
    budget_results: dict[int, dict[str, Any]] = {}

    # 在当前嵌套标签前缀上评分，禁止向模型传递尚未揭示的候选标签。
    def score_checkpoint(budget: int, complete: bool) -> None:
        rows, stats = score_evaluation_records(
            predictor, evaluation_records, acquired, budget, complete
        )
        group_id = hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[:16]
        for row in rows:
            row["group_id"] = group_id
        all_rows.extend(rows)
        budget_results[budget] = {
            "budget": budget,
            "labels_available": len(acquired),
            "complete": complete,
            "evaluation_responses": len(rows),
            **stats,
        }

    score_checkpoint(0, True)
    targets = set(BUDGETS[1:])
    for index, item_id in enumerate(candidate_ids):
        if len(acquired) >= max_labels:
            break
        items_remaining = len(candidate_ids) - index
        labels_remaining = max_labels - len(acquired)
        query_probability = min(1.0, labels_remaining / items_remaining)
        if rng.random() < query_probability:
            chosen = min(by_item[item_id], key=lambda record: record.response_id)
            acquired.append(chosen)
            if len(acquired) in targets:
                score_checkpoint(len(acquired), True)

    for budget in BUDGETS[1:]:
        if budget not in budget_results:
            score_checkpoint(budget, False)
    group_id = hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest()[:16]
    return GroupResult(
        group_id=group_id,
        benchmark_id=key[1],
        distinct_items=len(by_item),
        acquisition_items=len(acquisition_ids),
        evaluation_records=len(evaluation_records),
        budgets=budget_results,
    ), all_rows


# 将完整组的六档 Brier 按官方权重汇总；不足预算时明确拒绝伪造 ALC。
def group_alc(result: GroupResult) -> float | None:
    if any(not result.budgets[budget]["complete"] for budget in BUDGETS):
        return None
    return sum(
        ALC_WEIGHTS[index] * float(result.budgets[budget]["brier"])
        for index, budget in enumerate(BUDGETS)
    )


# 汇总各组等权的 Brier ALC，并分开报告不完整组以避免误作正式成绩。
def summarize_results(results: list[GroupResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("没有满足阈值的组可汇总")
    complete = [result for result in results if group_alc(result) is not None]
    per_budget: dict[str, dict[str, float | int | None]] = {}
    for budget in BUDGETS:
        eligible = [result for result in results if result.budgets[budget]["complete"]]
        per_budget[str(budget)] = {
            "complete_groups": len(eligible),
            "mean_brier": (
                sum(float(result.budgets[budget]["brier"]) for result in eligible)
                / len(eligible)
                if eligible
                else None
            ),
            "mean_ece": (
                sum(float(result.budgets[budget]["ece"]) for result in eligible)
                / len(eligible)
                if eligible
                else None
            ),
        }
    return {
        "protocol": {
            "labeled_visibility": "target_group_only",
            "protocol_note": "本地受控对照只向 predict 提供目标 subject–benchmark 组在当前预算下的标签前缀；不复刻官方全局调度中可能混入其他组标签的事件顺序。",
            "acquisition_response_selection": "first_by_response_id",
        },
        "budgets": list(BUDGETS),
        "alc_weights": list(ALC_WEIGHTS),
        "groups_total": len(results),
        "groups_complete": len(complete),
        "groups_incomplete": len(results) - len(complete),
        "mean_brier_alc_complete_groups": (
            sum(
                group_alc(result)
                for result in complete
                if group_alc(result) is not None
            )
            / len(complete)
            if complete
            else None
        ),
        "per_budget": per_budget,
        "groups": [
            {
                "group_id": result.group_id,
                "benchmark_id": result.benchmark_id,
                "distinct_items": result.distinct_items,
                "acquisition_items": result.acquisition_items,
                "evaluation_records": result.evaluation_records,
                "brier_alc": group_alc(result),
                "budgets": {str(budget): result.budgets[budget] for budget in BUDGETS},
            }
            for result in results
        ],
    }


# 写入逐响应 CSV，供之后独立复算分预算误差和校准诊断。
def write_response_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "group_id",
        "budget",
        "checkpoint_complete",
        "labels_available",
        "response_id",
        "item_id",
        "benchmark_id",
        "label",
        "prediction",
        "brier",
        "ece_bin",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# 执行本地评测并写入指定 runs 目录，输出不宣称为任何平台分数。
def evaluate_to_directory(
    records_path: Path,
    model_path: Path,
    output_dir: Path,
    seed: int,
    min_distinct_items: int,
) -> dict[str, Any]:
    if min_distinct_items < 2:
        raise ValueError("min_distinct_items 至少为 2")
    records = load_records(records_path)
    predictor = load_predictor(model_path)
    eligible_groups = {
        key: values
        for key, values in group_records(records).items()
        if len({record.item_id for record in values}) >= min_distinct_items
    }
    if not eligible_groups:
        raise ValueError("没有组达到 min_distinct_items 阈值")
    results: list[GroupResult] = []
    response_rows: list[dict[str, Any]] = []
    for key in sorted(eligible_groups):
        result, rows = evaluate_group(key, eligible_groups[key], predictor, seed)
        results.append(result)
        response_rows.extend(rows)
    summary = summarize_results(results)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "tool": "local_evaluation.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labeled_visibility": "target_group_only",
        "protocol_note": "本地公开数据诊断；不是 Codabench 平台成绩。只向 predict 提供目标 subject–benchmark 组在当前预算下的标签前缀，不复刻官方全局调度中可能混入其他组标签的事件顺序。",
        "records_path": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "seed": seed,
        "min_distinct_items": min_distinct_items,
        "split": "每个完整 subject–benchmark 组按 item 固定 50/50 acquisition/evaluation",
        "acquisition": "默认随机、无放回；每个 acquisition item 最多揭示一条响应",
        "acquisition_response_selection": "first_by_response_id",
        "budgets": list(BUDGETS),
        "alc_weights": list(ALC_WEIGHTS),
        "python": sys.version,
    }
    write_response_csv(output_dir / "per_response.csv", response_rows)
    (output_dir / "execution_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# 解析命令行参数，保持数据、模型、输出目录和随机种子显式可追溯。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="公开训练池导出的规范 JSONL response 记录",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="定义 predict(input, labeled) 的外部 model.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="新建的 runs 输出目录；必须不存在",
    )
    parser.add_argument(
        "--seed", type=int, default=20261014, help="item 划分与默认随机采样的固定种子"
    )
    parser.add_argument(
        "--min-distinct-items",
        type=int,
        default=80,
        help="参与评测的最小 distinct item 数，默认匹配官方少标签场景",
    )
    return parser.parse_args(argv)


# 提供可直接调用的 CLI 入口，并将摘要打印为 JSON 便于日志收集。
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate_to_directory(
        args.records, args.model, args.output_dir, args.seed, args.min_distinct_items
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
