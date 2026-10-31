"""Offline training entrypoint for the PAIEC cold-start logistic candidate."""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_evaluation import ResponseRecord, load_records, sha256_file

from coldstart.features import build_document, feature_schema

DEFAULT_MAX_FEATURES = 20_000
DEFAULT_PRIOR_STRENGTH = 2.0


# 以稳定 JSON 写入训练派生证据，保证配置和分割清单便于比较与复核。
def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


# 将规范化记录写成独立 JSONL，保留训练和留出数据的来源边界。
def write_records(path: Path, records: list[ResponseRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            raw = {
                "subject": record.subject,
                "item": record.item,
                "label": record.label,
                "response_id": record.response_id,
                "item_id": record.item_id,
                "benchmark_id": record.benchmark_id,
            }
            handle.write(
                json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


# 对完整 benchmark 留出划分训练记录，并显式断言两侧 benchmark 集合不相交。
def split_by_benchmark(
    records: list[ResponseRecord], heldout_benchmarks: set[str]
) -> tuple[list[ResponseRecord], list[ResponseRecord]]:
    train = [
        record for record in records if record.benchmark_id not in heldout_benchmarks
    ]
    heldout = [
        record for record in records if record.benchmark_id in heldout_benchmarks
    ]
    train_benches = {record.benchmark_id for record in train}
    heldout_benches = {record.benchmark_id for record in heldout}
    assert train_benches.isdisjoint(heldout_benches)
    if not train or not heldout:
        raise ValueError(
            "Both train and heldout benchmark partitions must contain records."
        )
    if heldout_benches != heldout_benchmarks:
        missing = sorted(heldout_benchmarks - heldout_benches)
        raise ValueError(f"Heldout benchmark(s) are absent from records: {missing}")
    return train, heldout


# 使用冻结 TF-IDF 与 liblinear 参数在训练分区拟合特征和二分类逻辑回归。
def fit_candidate(
    train_records: list[ResponseRecord], seed: int, max_features: int
) -> tuple[TfidfVectorizer, LogisticRegression]:
    labels = [record.label for record in train_records]
    if set(labels) != {0, 1}:
        raise ValueError("Training split must contain both binary classes 0 and 1.")
    documents = [
        build_document(record.subject, record.item) for record in train_records
    ]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        norm="l2",
        smooth_idf=True,
        sublinear_tf=False,
        min_df=1,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(documents)
    classifier = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=300,
        random_state=seed,
        class_weight=None,
    )
    classifier.fit(matrix, labels)
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Training classifier did not produce binary classes [0, 1].")
    return vectorizer, classifier


# 将 sklearn 参数压缩为仅标准库推理需要的 JSON artifact，不序列化训练对象或标签。
def make_artifact(
    vectorizer: TfidfVectorizer, classifier: LogisticRegression, prior_strength: float
) -> dict[str, object]:
    if not prior_strength > 0:
        raise ValueError("prior_strength must be positive.")
    return {
        "artifact_version": "paiec-coldstart-logistic-v1",
        "feature_schema": feature_schema(),
        "vocabulary_": {
            term: int(index) for term, index in vectorizer.vocabulary_.items()
        },
        "idf_": vectorizer.idf_.astype(float).tolist(),
        "coef_": classifier.coef_[0].astype(float).tolist(),
        "intercept_": float(classifier.intercept_[0]),
        "classes_": [0, 1],
        "prior_strength": float(prior_strength),
    }


# 汇总输入、源码与库版本哈希，记录实际训练来源而不读取或伪造额外数据。
def provenance(records_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        root / "coldstart" / "model.py",
        root / "coldstart" / "features.py",
        Path(__file__),
    )
    return {
        "input_sha256": sha256_file(records_path),
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in source_paths
        },
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


# 执行一次离线训练并写入新输出目录中的 split、配置、artifact 与追溯信息。
def train(
    records_path: Path,
    heldout_benchmarks: set[str],
    output_dir: Path,
    seed: int,
    max_features: int,
    prior_strength: float,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive.")
    records = load_records(records_path)
    train_records, heldout_records = split_by_benchmark(records, heldout_benchmarks)
    vectorizer, classifier = fit_candidate(train_records, seed, max_features)
    output_dir.mkdir(parents=True)
    write_records(output_dir / "train.jsonl", train_records)
    write_records(output_dir / "heldout.jsonl", heldout_records)
    split = {
        "heldout_benchmarks": sorted(heldout_benchmarks),
        "train_benchmarks": sorted({record.benchmark_id for record in train_records}),
        "heldout_records": len(heldout_records),
        "train_records": len(train_records),
    }
    write_json(output_dir / "split.json", split)
    artifact = make_artifact(vectorizer, classifier, prior_strength)
    write_json(output_dir / "artifact.json", artifact)
    config = {
        "seed": seed,
        "max_features": max_features,
        "prior_strength": prior_strength,
        "vectorizer": {
            "analyzer": "word",
            "lowercase": True,
            "token_pattern": "(?u)\\b\\w\\w+\\b",
            "norm": "l2",
            "smooth_idf": True,
            "sublinear_tf": False,
            "min_df": 1,
            "max_features": max_features,
        },
        "classifier": {
            "solver": "liblinear",
            "C": 1.0,
            "max_iter": 300,
            "random_state": seed,
            "class_weight": None,
            "calibration": "not applied",
        },
        "feature_schema": feature_schema(),
    }
    write_json(output_dir / "config.json", config)
    manifest = {
        "provenance": provenance(records_path),
        "split": split,
        "artifact_sha256": sha256_file(output_dir / "artifact.json"),
        "status": "Offline training artifact only; no calibration or effectiveness claim.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return {
        "artifact": artifact,
        "classifier": classifier,
        "vectorizer": vectorizer,
        "split": split,
    }


# 解析命令行参数并调用离线训练，不下载数据、不调用 API 或上传任何材料。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, help="Authorized JSONL records path"
    )
    parser.add_argument(
        "--heldout-benchmark",
        action="append",
        required=True,
        help="Benchmark ID reserved entirely for holdout; repeat as needed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for local artifacts",
    )
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    args = parser.parse_args()
    train(
        records_path=args.records,
        heldout_benchmarks=set(args.heldout_benchmark),
        output_dir=args.output_dir,
        seed=args.seed,
        max_features=args.max_features,
        prior_strength=args.prior_strength,
    )
    print(f"Wrote offline cold-start artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
