# PAIEC S4 predictor

系统配置先验与稀疏标签更新的轻量预测器。当前仓库整理的是 **S4 候选**：
它先从允许的系统配置估计成功概率，再利用同组已揭示标签进行固定 spike–Beta 更新。
推理只使用 Python 标准库，不调用大模型接口，不需要 GPU。

这是本地 Git 仓库，尚未发布到 GitHub。原始研究工作区和已冻结材料保留。

## 先看哪些代码

| 文件 | 作用 |
| --- | --- |
| [runtime/model.py](runtime/model.py) | 概率预测、共享校准、当前组标签过滤与后验更新 |
| [runtime/features.py](runtime/features.py) | 完整可见 subject 字典的稳定序列化 |
| [runtime/artifact.json](runtime/artifact.json) | 冻结词表、IDF、系数与混合参数 |
| [runtime/calibration.json](runtime/calibration.json) | 共享 logit 偏移和适用 provider 集合 |
| [tools/reproduce_final_s4_head.py](tools/reproduce_final_s4_head.py) | 从四个公开训练任务重建最终配置头 |
| [tools/fit_s4_shared_calibration.py](tools/fit_s4_shared_calibration.py) | 从任务留出预测拟合共享校准 |
| [tools/local_evaluation.py](tools/local_evaluation.py) | 组、标签预算与 Brier ALC 的本地计算 |
| [release.py](release.py) | 核对源码、生成输入清单和打包预测器 |

## 推理示例

在仓库根目录运行下列合成输入示例。不读取真实数据，不产生 API 费用：

```python
import sys
sys.path.insert(0, "runtime")
from model import predict

subject = {"normalized_name": "example-system", "provider": "example-provider"}
item = {"benchmark_id": "example-benchmark", "id": "query"}
p0 = predict([subject, item], [])
labeled = [[[subject, {"benchmark_id": "example-benchmark", "id": "seen"}], 1]]
p1 = predict([subject, item], labeled)
print(p0, p1)
```

预测器只使用 subject 字段及同 subject、同匿名 benchmark 的当前标签。
对固定标签历史，同组不同题目得到相同概率；它不建模逐题难度。

## 固定候选与证据范围

`runtime/` 的四个文件逐字节对应 `artifacts/runtime.zip`，即已有提交 **994883** 的 S4 运行包。
该次形成性评测显示 Brier ALC **0.164554**。这不是最终共同测试成绩或排名；
不同形成性提交重采样隐藏数据，分差不能解释为配对改进。

该运行头由四任务的 177,977 条有效响应、252 个组拟合。
本文五任务开发比较使用独立的留任务训练头，不能用这个全 S4 头在其训练任务上冒充留出评测。
仓库覆盖固定候选的推理、数据准备、最终头拟合和校准复现链。
`tools/run_*` 是该链调用的历史研究模块；它们的独立实验入口还依赖原工作区的数据布局。
本仓库不声称已封装全部历史实验，也不把下列文件检查称为重新训练或效果复验。

## 本地核对与封包

```text
python -I -B release.py verify-source
python -I -B release.py pack-runtime --artifact runtime/artifact.json --calibration runtime/calibration.json --output runtime_copy.zip
```

第二条命令要求输出文件尚不存在。它核对冻结参数后封装同一组四成员；
成员身份应与已提交包一致，ZIP 容器的字节可能因压缩环境不同而变化。
直接需要原始提交对象时使用 `artifacts/runtime.zip`。

## 从合法取得的训练输入复现

先按 [INPUT_REQUIREMENTS.json](INPUT_REQUIREMENTS.json) 中的固定 MDB revision 和文件清单取得数据。
仓库只包含 allowlist 与输入身份，不附带原始题文、标签或响应表，也不自动下载。
训练环境版本见 [requirements.txt](requirements.txt)；原运行使用 Python 3.12.3。
安装依赖、获取数据和完整训练应在自己的已批准资源范围内执行。

<details>
<summary>数据准备、校准与训练命令</summary>

在仓库根目录执行，各输出目录应为新目录：

```text
python tools/prepare_public_data.py --data-root payload/public_raw/r1 --allowlist payload/public_raw/r1/allowlist.json --output reproduction/normalized/r1/records.jsonl --manifest reproduction/normalized/r1/manifest.json
python tools/prepare_public_data.py --data-root payload/public_raw/r2_statqa --allowlist payload/public_raw/r2_statqa/allowlist.json --output reproduction/normalized/r2_statqa/records.jsonl --manifest reproduction/normalized/r2_statqa/manifest.json
python tools/prepare_public_data.py --data-root payload/public_raw/r3_truthfulqa --allowlist payload/public_raw/r3_truthfulqa/allowlist.json --output reproduction/normalized/r3_truthfulqa/records.jsonl --manifest reproduction/normalized/r3_truthfulqa/manifest.json
python tools/reproduce_r1_partitions.py --records reproduction/normalized/r1/records.jsonl --records-sha256 1bdafeaa19dd77253c517bd0c1e9147a5d8312415ed8b5a4321b324c49af3e0a --output-root reproduction/r1_partitions
python -I -B release.py make-input-manifest --prepared-root reproduction --output reproduction/s4_input_manifest.json
python tools/fit_s4_shared_calibration.py --input-manifest reproduction/s4_input_manifest.json --output-root reproduction/calibration --cap-seconds 20
python tools/reproduce_final_s4_head.py --input-manifest reproduction/s4_input_manifest.json --output-root reproduction/head
python -I -B release.py pack-runtime --artifact reproduction/head/artifact.json --calibration reproduction/calibration/calibration.json --output reproduction/runtime.zip
```

输入字节身份、最终头参数和校准语义均有固定检查。原规范化环境使用 Windows 文本换行，
R1 分任务输出为 LF；换行或库版本变化可能使字节检查停止，不能据此放宽输入要求。
历史运行已经完成分阶段复现，本次仓库整理没有重新执行训练。

</details>

## 来源与许可

[NOTICE.txt](NOTICE.txt) 记录当前许可状态；[provenance/source_origins.json](provenance/source_origins.json)
对应实际执行源码快照。没有为第三方材料或本项目自动套用 MIT 等许可证。
公开前需确定发布账号、具体内容与权利人的许可选择。
