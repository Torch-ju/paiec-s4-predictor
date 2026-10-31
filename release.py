"""Verify and package the frozen S4 predictor; no network or training is started."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SOURCE_MANIFEST_NAME = "SOURCE_MANIFEST.json"

EXPECTED_ARTIFACT_SHA256 = (
    "bc6961b07044c097c40f6dd2af2c1c4e76a265be16678df563627d0d0f4bfd8e"
)

EXPECTED_CALIBRATION_SEMANTIC_SHA256 = (
    "df5552a162ab7761357133060b8ee016bc844f9fddf518b3c9c55957b73a173b"
)

TASKS = ("researchcodebench", "real_webagents", "statqa", "truthfulqa_mc")

PREPARED_RELATIVE_PATHS = (
    "r1_partitions/researchcodebench/heldout.jsonl",
    "r1_partitions/real_webagents/heldout.jsonl",
    "normalized/r2_statqa/records.jsonl",
    "normalized/r3_truthfulqa/records.jsonl",
)

PREPARED_SHA256 = (
    "53394dc05debf69096ec3210e0570f54becb432b3405f815250e66a43fde067b",
    "c537a5c1297303c088ddc4e7dff1991f0b5b3275d31313f104074445fbd2d1c2",
    "9302059479242986f3fbd56d1fb30b269828b967800ef97259008f270b8302e6",
    "fb155648730d487687ef00b30854206261b35d920341e9def042c6aece8ee846",
)

RUNTIME_MEMBERS = ("model.py", "features.py", "artifact.json", "calibration.json")

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


# 计算文件字节身份，用于源码和参数一致性核验。


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


# 对规范化 JSON 求哈希，核对校准参数的数学语义。


def stable_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# 拒绝绝对路径及上级跳转，限定仓库成员范围。


def safe_relative_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or name != path.as_posix()
    ):
        raise ValueError(f"Unsafe proposed member path: {name!r}")
    return path


# 固定 ZIP 成员时间和权限，使相同成员可重复封装。


def archive_entry(name: str) -> ZipInfo:
    entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = ZIP_DEFLATED
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    return entry


# 写出可读且排序稳定的 JSON 记录。


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


# 逐项核对仓库清单中的大小和哈希，不运行训练或网络请求。


def verify_source(package_root: Path) -> int:
    manifest_path = package_root / SOURCE_MANIFEST_NAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema") != "paiec-s4-method-source-review-v1"
    ):
        raise ValueError(
            "Source manifest schema differs from the local review contract."
        )
    rows = document.get("copied_members")
    generated = document.get("generated_members")
    if not isinstance(rows, list) or not isinstance(generated, list):
        raise TypeError("Source manifest member lists are invalid.")
    seen: set[str] = set()
    for row in [*rows, *generated]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Source manifest member identity is invalid.")
        name, size, digest = row["path"], row["bytes"], row["sha256"]
        relative = safe_relative_path(name)
        if name in seen or not isinstance(size, int) or not isinstance(digest, str):
            raise ValueError(
                "Source manifest contains duplicate or malformed member identity."
            )
        seen.add(name)
        path = (package_root / relative).resolve()
        if (
            package_root.resolve() not in path.parents
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError(f"Source manifest member is unsafe or absent: {name}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"Source manifest member identity mismatch: {name}")
    return len(seen)


# 要求新输出和既有父目录，避免覆盖历史实验。


def validate_new_output(output: Path) -> Path:
    resolved = output.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {resolved}")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory does not exist: {resolved.parent}"
        )
    return resolved


# 核验四任务规范化输入后生成当前机器上的显式路径清单。


def make_input_manifest(prepared_root: Path, output: Path) -> None:
    root = prepared_root.resolve()
    output = validate_new_output(output)
    if not root.is_dir() or output.parent.resolve() != root:
        raise ValueError("--output must be a new file directly inside --prepared-root.")
    try:
        root.relative_to(active_package_root().resolve())
    except ValueError as exc:
        raise ValueError(
            "--prepared-root must remain inside the unpacked package root."
        ) from exc
    inputs: list[dict[str, str]] = []
    for task, relative, expected_sha in zip(
        TASKS, PREPARED_RELATIVE_PATHS, PREPARED_SHA256, strict=True
    ):
        path = (root / PurePosixPath(relative)).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Missing prepared input for {task}: {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"Prepared input SHA-256 mismatch for {task}.")
        inputs.append(
            {
                "task": task,
                "kind": "response_jsonl",
                "path": str(path),
                "sha256": actual_sha,
            }
        )
    write_json(output, {"schema_version": "paiec-s4-oof-inputs-v1", "inputs": inputs})


# 核对固定共享偏移和 provider 集合，允许复现来源元信息不同。


def validate_calibration(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "calibration_version",
        "method",
        "theta",
        "seen_providers",
        "source_summary_sha256",
        "source_input_manifest_sha256",
        "cmmlu_labels_in_shared_theta",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("Calibration schema differs from the S4 runtime contract.")
    if (
        document["calibration_version"] != "paiec-provider-agnostic-masked-shared-v1"
        or document["method"] != "provider-agnostic-shared-logit-offset"
        or document["cmmlu_labels_in_shared_theta"] is not False
        or not isinstance(document["theta"], (int, float))
        or isinstance(document["theta"], bool)
        or not isinstance(document["seen_providers"], list)
        or any(
            not isinstance(item, str) or not item for item in document["seen_providers"]
        )
        or len(document["seen_providers"]) != len(set(document["seen_providers"]))
    ):
        raise ValueError("Calibration runtime fields are invalid.")
    for field in ("source_summary_sha256", "source_input_manifest_sha256"):
        if not isinstance(document[field], str) or not SHA256_PATTERN.fullmatch(
            document[field]
        ):
            raise ValueError(f"Calibration source metadata is invalid: {field}")
    semantic = {
        field: document[field]
        for field in (
            "calibration_version",
            "method",
            "theta",
            "seen_providers",
            "cmmlu_labels_in_shared_theta",
        )
    }
    if stable_sha256(semantic) != EXPECTED_CALIBRATION_SEMANTIC_SHA256:
        raise ValueError(
            "Calibration semantics differ from the verified S4 method identity."
        )
    return document


# 仅从明确给出的固定参数封装四成员预测 ZIP。


def pack_runtime(artifact: Path, calibration: Path, output: Path) -> None:
    package_root = active_package_root()
    verify_source(package_root)
    output = validate_new_output(output)
    if not artifact.is_file() or sha256_file(artifact) != EXPECTED_ARTIFACT_SHA256:
        raise ValueError(
            "Artifact identity differs from the verified final-head contract."
        )
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration file is missing: {calibration}")
    validate_calibration(calibration)
    runtime = package_root / "runtime"
    files = {
        "model.py": runtime / "model.py",
        "features.py": runtime / "features.py",
        "artifact.json": artifact,
        "calibration.json": calibration,
    }
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError(
            "Runtime source or explicit new parameter output is missing."
        )
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in RUNTIME_MEMBERS:
            archive.writestr(archive_entry(name), files[name].read_bytes())


# 返回当前独立仓库根目录，不依赖原始工作区或历史打包目录。


def active_package_root() -> Path:
    return Path(__file__).resolve().parent


# 解析验证、输入清单与打包三个本地动作；不提供发布或批量实验入口。


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-source")
    manifest = commands.add_parser("make-input-manifest")
    manifest.add_argument("--prepared-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    pack = commands.add_parser("pack-runtime")
    pack.add_argument("--artifact", type=Path, required=True)
    pack.add_argument("--calibration", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


# 执行显式本地命令并返回状态，不隐式训练、安装依赖或上传。


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify-source":
        print(f"Verified {verify_source(active_package_root())} repository members.")
    elif args.command == "make-input-manifest":
        make_input_manifest(args.prepared_root, args.output)
        print(f"Wrote verified input manifest: {args.output}")
    else:
        pack_runtime(args.artifact, args.calibration, args.output)
        print(f"Wrote runtime ZIP: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
