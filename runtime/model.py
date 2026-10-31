"""Standard-library inference for the frozen visible-subject mixture artifact."""

import json
import math
import numbers
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from features import FEATURE_SCHEMA_VERSION, build_document

ARTIFACT_FILENAME = "artifact.json"
CALIBRATION_FILENAME = "calibration.json"
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


# 读取并严格核验同目录 artifact，只缓存冻结参数而不缓存任何调用标签。
@lru_cache(maxsize=1)
def _load_artifact() -> dict[str, Any]:
    path = Path(__file__).with_name(ARTIFACT_FILENAME)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required cold-start artifact: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read cold-start artifact: {path}") from exc
    if not isinstance(artifact, dict):
        raise TypeError("Cold-start artifact must be a JSON object.")
    if artifact.get(
        "artifact_version"
    ) != "paiec-visible-subject-mixture-v1" or artifact.get("adaptation") != {
        "method": "fixed-spike-and-slab",
        "spike_mass": 0.5,
        "slab_strength": 2.0,
    }:
        raise ValueError(
            "Incompatible frozen mixture artifact identity or adaptation parameters."
        )
    schema = artifact.get("feature_schema")
    if not isinstance(schema, dict) or schema.get("version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Cold-start artifact has an incompatible feature schema.")
    vocabulary = artifact.get("vocabulary_")
    idf = artifact.get("idf_")
    coef = artifact.get("coef_")
    if (
        not isinstance(vocabulary, dict)
        or not isinstance(idf, list)
        or not isinstance(coef, list)
    ):
        raise TypeError(
            "Cold-start artifact is missing vectorizer or classifier parameters."
        )
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("Cold-start artifact vocabulary indices must be contiguous.")
    if len(idf) != len(vocabulary) or len(coef) != len(vocabulary):
        raise ValueError("Cold-start artifact vector dimensions do not agree.")
    if artifact.get("classes_") != [0, 1]:
        raise ValueError("Cold-start artifact must encode binary classes [0, 1].")
    for name, values in (("idf_", idf), ("coef_", coef)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"Cold-start artifact {name} contains non-finite values.")
    for name in ("intercept_", "prior_strength"):
        value = artifact.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Cold-start artifact {name} must be finite.")
    if float(artifact["prior_strength"]) <= 0:
        raise ValueError("Cold-start artifact prior_strength must be positive.")
    return artifact


# 读取同目录共享校准，只接受已冻结的 provider 无关 theta 与精确可见 provider 集合。
@lru_cache(maxsize=1)
def _load_calibration() -> dict[str, Any]:
    path = Path(__file__).with_name(CALIBRATION_FILENAME)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required masked calibration: {path}")
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read masked calibration: {path}") from exc
    if not isinstance(calibration, dict):
        raise TypeError("Masked calibration must be a JSON object.")
    if (
        calibration.get("calibration_version")
        != "paiec-provider-agnostic-masked-shared-v1"
    ):
        raise ValueError("Incompatible masked calibration version.")
    if calibration.get("method") != "provider-agnostic-shared-logit-offset":
        raise ValueError("Incompatible masked calibration method.")
    theta = calibration.get("theta")
    providers = calibration.get("seen_providers")
    if (
        isinstance(theta, bool)
        or not isinstance(theta, numbers.Real)
        or not math.isfinite(float(theta))
    ):
        raise ValueError("Masked calibration theta must be finite.")
    if (
        not isinstance(providers, list)
        or not providers
        or any(not isinstance(provider, str) or not provider for provider in providers)
        or len(providers) != len(set(providers))
    ):
        raise ValueError("Masked calibration providers must be a unique string list.")
    return {"theta": float(theta), "seen_providers": frozenset(providers)}


# 以 sklearn 默认 word token 规则重建词频，保持训练时 lowercase 和 token_pattern 语义。
def _term_counts(document: str) -> Counter[str]:
    return Counter(TOKEN_PATTERN.findall(document.lower()))


# 对 count*idf 向量做 L2 归一化并计算二分类线性得分，不依赖 sklearn 运行时。
def _base_probability(
    subject: dict[str, Any], item: dict[str, Any], artifact: dict[str, Any]
) -> float:
    vocabulary: dict[str, int] = artifact["vocabulary_"]
    idf: list[float] = artifact["idf_"]
    coefficients: list[float] = artifact["coef_"]
    weighted: list[tuple[int, float]] = []
    for token, count in _term_counts(build_document(subject, item)).items():
        index = vocabulary.get(token)
        if index is not None:
            weighted.append((index, count * float(idf[index])))
    norm = math.sqrt(sum(value * value for _, value in weighted))
    score = float(artifact["intercept_"])
    if norm:
        score += sum(
            value / norm * float(coefficients[index]) for index, value in weighted
        )
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp_score = math.exp(score)
    return exp_score / (1.0 + exp_score)


# 仅按完整可见 subject 的 provider 精确成员关系应用共享 logit 偏移；空或未见值保持原先验。
def _masked_provider_prior(
    prior: float, subject: dict[str, Any], calibration: dict[str, Any]
) -> float:
    provider = subject.get("provider")
    if not isinstance(provider, str) or provider not in calibration["seen_providers"]:
        return prior
    log_odds = math.log(prior) - math.log1p(-prior) + calibration["theta"]
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_value = math.exp(log_odds)
    return exp_value / (1.0 + exp_value)


# 校验后验计数与先验参数，拒绝布尔值、非有限值和无效概率边界。
def _validate_parameters(
    successes: object,
    count: object,
    prior: object,
    spike_mass: object,
    slab_strength: object,
) -> tuple[int, int, float, float, float]:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be a non-boolean integer")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be a non-boolean integer")
    if successes < 0 or count < 0 or successes > count:
        raise ValueError("successes must satisfy 0 <= successes <= count")
    values = (prior, spike_mass, slab_strength)
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Real)
        for value in values
    ):
        raise TypeError("prior, spike_mass, and slab_strength must be real numbers")
    q, pi, kappa = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (q, pi, kappa)):
        raise ValueError("parameters must be finite")
    if not 0.0 < q < 1.0 or not 0.0 <= pi <= 1.0 or kappa <= 0.0:
        raise ValueError(
            "prior must be strictly inside (0, 1); spike_mass must be a probability and slab_strength positive"
        )
    return successes, count, q, pi, kappa


# 用稳定 logistic 把有限或极端 log-odds 转成 [0,1] 内的 slab 后验权重。
def _stable_logistic(log_odds: float) -> float:
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exp_value = math.exp(log_odds)
    return exp_value / (1.0 + exp_value)


# 计算 spike 与 Beta slab 的对数边际似然差，不在 q 边界调用 log 或 lgamma(0)。
def _log_likelihood_difference(
    successes: int, count: int, prior: float, strength: float
) -> float:
    failures = count - successes
    log_spike = successes * math.log(prior) + failures * math.log1p(-prior)
    alpha = strength * prior
    beta = strength * (1.0 - prior)
    log_slab = (
        math.lgamma(alpha + successes)
        + math.lgamma(beta + failures)
        - math.lgamma(strength + count)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(strength)
    )
    return log_slab - log_spike


# 返回严格内部先验的 spike-and-slab Beta 后验均值，拒绝零概率证据的退化端点。
def posterior_probability(
    successes: int,
    count: int,
    prior: float,
    spike_mass: float = 0.5,
    slab_strength: float = 2,
) -> float:
    successes, count, prior, spike_mass, slab_strength = _validate_parameters(
        successes, count, prior, spike_mass, slab_strength
    )
    if count == 0:
        return prior
    slab_mean = (successes + slab_strength * prior) / (count + slab_strength)
    if spike_mass == 0.0:
        return slab_mean
    if spike_mass == 1.0:
        return prior
    log_odds = _log_likelihood_difference(successes, count, prior, slab_strength)
    log_odds += math.log1p(-spike_mass) - math.log(spike_mass)
    slab_weight = _stable_logistic(log_odds)
    return (1.0 - slab_weight) * prior + slab_weight * slab_mean


# 解析正标签预算中的单条观测，显式拒绝缺失 benchmark 元数据和非法二元响应。
def _validated_observation(
    observation: object,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    try:
        observed_input, label = observation  # type: ignore[misc]
        observed_subject, observed_item = observed_input
    except (TypeError, ValueError):
        raise TypeError(
            "Each acquired observation must be [[subject, item], binary_label]."
        ) from None
    if not isinstance(observed_subject, dict) or not isinstance(observed_item, dict):
        raise TypeError(
            "Each acquired observation must include subject and item dictionaries."
        )
    benchmark = observed_item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError(
            "Acquired items must include a nonempty anonymous benchmark_id."
        )
    if (
        isinstance(label, bool)
        or not isinstance(label, numbers.Real)
        or not math.isfinite(float(label))
        or label not in (0, 1)
    ):
        raise ValueError("Acquired responses must be binary numeric labels (0 or 1).")
    return observed_subject, observed_item, int(label)


# 计算冻结配置先验，仅用当前同组标签实施固定spike-and-slab后验平均。
def predict(input: list[dict[str, Any]], labeled: list | None = None) -> float:
    """Predict from the frozen sibling artifact without network or cross-call label state."""
    try:
        subject, item = input
    except (TypeError, ValueError):
        raise ValueError("input must be [subject, item].") from None
    if not isinstance(subject, dict) or not isinstance(item, dict):
        raise TypeError("input must contain subject and item dictionaries.")
    artifact = _load_artifact()
    prior = _masked_provider_prior(
        _base_probability(subject, item, artifact), subject, _load_calibration()
    )
    if not labeled:
        return prior
    benchmark = item.get("benchmark_id")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("Positive label budgets require target item benchmark_id.")

    successes = 0
    observations = 0
    for observation in labeled:
        observed_subject, observed_item, label = _validated_observation(observation)
        if observed_subject != subject or observed_item["benchmark_id"] != benchmark:
            continue
        successes += label
        observations += 1
    strength = float(artifact["prior_strength"])
    return posterior_probability(successes, observations, prior, 0.5, strength)
