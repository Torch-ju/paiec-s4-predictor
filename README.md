# PAIEC S4 Predictor

S4 is a group-level probability predictor developed for the Predictive AI Evaluation Competition. Given an AI system configuration and the labels revealed for a subject–benchmark group, it estimates the probability that the system will answer a target correctly. The predictor does not run the target system.

A task-weighted logistic model produces a configuration-derived prior. An out-of-fold, provider-masked correction adjusts that prior, and a fixed spike–Beta update incorporates the revealed labels. At inference, the predictor uses only Python's standard library; it does not call a model API or require a GPU.

## Repository guide

| Path | Purpose |
| --- | --- |
| [runtime/model.py](runtime/model.py) | Prediction, calibration, group-label filtering and posterior update |
| [runtime/features.py](runtime/features.py) | Stable serialization of the visible subject configuration |
| [runtime/artifact.json](runtime/artifact.json) | Frozen vocabulary, IDF values, coefficients and mixture parameters |
| [runtime/calibration.json](runtime/calibration.json) | Frozen shared correction and provider set |
| [artifacts/runtime.zip](artifacts/runtime.zip) | Original four-file Codabench submission archive |
| [tools/reproduce_final_s4_head.py](tools/reproduce_final_s4_head.py) | Reconstruct the four-task final configuration head |
| [tools/fit_s4_shared_calibration.py](tools/fit_s4_shared_calibration.py) | Fit the shared calibration from out-of-fold predictions |
| [tools/local_evaluation.py](tools/local_evaluation.py) | Local group-level Brier ALC calculation |
| [release.py](release.py) | Verify source files, create an input manifest and package the runtime |

The four files in `runtime/` are byte-identical to the members of `artifacts/runtime.zip`, the archive associated with Codabench submission **994883**. The ZIP is retained as the original submitted archive.

## Run the predictor

From the repository root, the following synthetic example calls `predict` without training data or network access:

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

The first input describes the subject and the second describes the target item. `labeled` contains the outcomes revealed by the evaluation interface. Only labels for the same subject and anonymous benchmark are used in the update. The output is a probability in `[0, 1]`.

The predictor does not use item text or item features. For a fixed subject–benchmark group and revealed-label history, it assigns the same probability to every item; it does not model item-specific difficulty.

## Evaluation scope

The final runtime head was fitted on four public training tasks, with 177,977 valid responses across 252 groups. The five-task development study uses separate leave-one-task-out heads; the final four-task head must not be treated as a held-out model on tasks used to fit it.

Formative submissions use resampled hidden subsets. Their scores are not paired comparisons and do not establish a final common-test result or rank. The repository supports inference and reconstruction of the final head and calibration; it does not claim to package every historical experiment, and private formative outcomes cannot be independently reconstructed from these files.

## Verify and package

From the repository root:

```text
python -I -B release.py verify-source
python -I -B release.py pack-runtime --artifact runtime/artifact.json --calibration runtime/calibration.json --output runtime_copy.zip
```

The `pack-runtime` output path must not already exist. It checks the frozen parameters and packages the same four runtime members. ZIP container bytes may vary between compression environments; use `artifacts/runtime.zip` when the original submitted archive is required.

## Reconstruct the fitted runtime

The raw question text, labels and response tables are not included or downloaded automatically. Obtain the permitted MeasurementDB inputs identified in [INPUT_REQUIREMENTS.json](INPUT_REQUIREMENTS.json), including its pinned revision, paths and SHA-256 checksums. Place them under the indicated `payload/public_raw/` paths. The repository includes the corresponding allowlists.

The recorded training run used Python 3.12.3. The preparation and training dependencies are listed in [requirements.txt](requirements.txt); they are not needed for ordinary runtime inference. Run the commands below from the repository root with fresh output directories:

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

The pipeline checks input byte identities and the final head and calibration semantics. The recorded normalization used Windows text newlines, while the R1 task partitions used LF. Newline or library-version differences can stop byte-level checks; they should be investigated rather than bypassed. These commands describe the documented reconstruction workflow, not a newly executed experiment.

## Sources and rights

[NOTICE.txt](NOTICE.txt) records the attribution and license status; [provenance/source_origins.json](provenance/source_origins.json) identifies source snapshots. Provenance records preserve the context of earlier reviews; historical references to publication status or archive identity do not describe the current hosting status. No project-wide license is granted by this README. Third-party components and data remain subject to their own terms.
