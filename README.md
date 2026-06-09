# PP-DMA: Privacy-Preserving Distributed Model Adaptation

Group 4 final project for **Distributed Computing Systems**.

The original topic was **Asynchronous Distributed ML Adaptation**.  The final
project now focuses on a broader distributed-systems question:

> In privacy-preserving federated adaptation without a global clock, can an
> asynchronous method approach Sync FedAvg while controlling stale updates,
> stragglers, and fast-client domination?

The repository contains two connected tracks:

1. **MedMNIST federated image classification**: the main algorithmic research
   track with multi-dataset, multi-seed experiments.
2. **LLM/MLLM medical QA/VQA DEMO**: a demo-oriented extension showing how the
   same clockless aggregation idea can be applied to LoRA / QLoRA adapter
   updates for medical closed-ended QA and VQA.

This is a course-project research/demo codebase, not a production FL platform.

## Team

| Name | Student ID | Department |
| --- | --- | --- |
| 陳冠宇 | R13946001 | Data Science |
| 張光澄 | R14922172 | Computer Science |
| 張育嘉 | R14922140 | Computer Science |

## Why This Project

Medical AI is naturally distributed:

- hospitals cannot freely centralize raw medical images or clinical questions;
- hospitals have different GPUs, workloads, network delays, and patient
  distributions;
- synchronous training is stable but waits for slow clients;
- asynchronous training improves throughput but introduces stale and conflicting
  updates.

We model the system without a synchronized physical clock.  Instead, the server
uses logical model versions:

```text
staleness = current_server_version - client_start_version
```

The distributed-systems focus is therefore not just model accuracy, but also
event ordering, staleness, straggler behavior, client contribution imbalance,
and whether async training can remain close to Sync FedAvg under a fair update
budget.

## Core Methods

| Method | Role | Main idea |
| --- | --- | --- |
| Sync FedAvg | baseline | Barrier synchronization; stable but waits for all clients. |
| Naive Async | baseline | Applies each arriving update immediately. |
| Staleness Async | baseline | Downweights updates using logical staleness. |
| FedBuff-lite | baseline | Buffers async updates before aggregation. |
| CAA-FedBuff | proposed v1 | Adds direction agreement, clipping, and adaptive alpha. |
| CAA-v2 | final proposed method | Adds server trajectory agreement and client fairness credit. |

CAA-v2 is our implemented design extension.  It combines known ideas from
buffered async FL, staleness-aware weighting, cosine agreement, delta clipping,
adaptive server step size, and client fairness into a simple clockless rule:

```text
raw_weight_i =
    num_examples_i
  * staleness_decay(tau_i)
  * agreement_factor_i
  * fairness_credit_i

tau_i = current_server_version - client_start_version_i
```

We claim this as a clear system-integration contribution for a distributed
systems course project, not as a publication-level SOTA FL algorithm.

## LLM / MLLM Extension

The LLM/MLLM DEMO uses the same CAA-v2 idea at the adapter level:

```text
delta_i = local_adapter_i - global_adapter_at_client_start
```

Instead of aggregating a full 3B model, the system freezes the base model and
aggregates only LoRA / QLoRA adapter deltas.  This keeps communication and GPU
memory realistic for a demo setting.

The closed-ended QA/VQA setup uses finite answer spaces such as `A/B/C/D` or
`yes/no`, so results can be evaluated with accuracy and invalid-answer rate.

## Final DEMO Package

The clean handoff folder is:

```text
PP-DMA_Demo_Final_Package/
```

Start here:

```text
PP-DMA_Demo_Final_Package/slides/PP-DMA_Demo_Final.pptx
PP-DMA_Demo_Final_Package/docs/LLM_MLLM_DEMO_REFERENCE.md
PP-DMA_Demo_Final_Package/docs/demo_final_speaker_notes_zh.md
```

The package also contains selected figures, compact CSV summaries, and paired
VQA examples.  Raw logs, checkpoints, full results, and datasets are excluded.

## Main DEMO Result

Peer Qwen3-VL + PMC-VQA pilot result:

| Method | Correct / Total | Accuracy |
| --- | ---: | ---: |
| Sync FedAvg | 47 / 100 | 47.0% |
| Naive Async | 46 / 100 | 46.0% |
| Staleness Async | 45 / 100 | 45.0% |
| FedBuff | 44 / 100 | 44.0% |
| CAA-v2 | 47 / 100 | 47.0% |

Conservative DEMO claim:

> In the Qwen3-VL + PMC-VQA pilot, CAA-v2 ties Sync FedAvg and is the strongest
> async-family method in that run.

Local LLM/MLLM support:

- Text MCQA: CAA-v2 has the highest mean best accuracy, while Sync remains
  slightly better in final stability.
- Real Qwen2.5-VL 3B 4-bit QLoRA: closed-answer valid rate is 100% in the small
  feasibility diagnostic.
- Paired VQA diagnostics: selected same-question examples show CAA-v2 correcting
  some Naive Async or Sync errors.  We do **not** claim a base-wrong /
  CAA-correct before-after improvement when no exact paired case was found.

## What the Results Mean: Sync vs Async

The results should be read as a distributed-systems tradeoff, not only an
accuracy contest.

- **Sync FedAvg is the safest baseline** when waiting is acceptable.  It is
  stable because the server uses a round barrier.
- **Naive Async removes the barrier**, so it better matches real heterogeneous
  systems, but stale or conflicting updates can hurt stability.
- **Staleness-only async is safe but often too conservative**, because it only
  looks at logical age and ignores whether an update direction is useful.
- **CAA-v2 is the project answer**: it keeps the asynchronous, no-waiting
  setting, but adds clockless agreement and fairness signals so async behavior
  can approach Sync FedAvg more reliably.

This connects back to the motivation: hospitals should not have to centralize
data or wait for every slow site forever.  The practical distributed benefit is
not that async always beats sync; it is that async can reduce barrier waiting
while CAA-v2 helps control the correctness cost of stale updates.

## MedMNIST Research Track

The MedMNIST track is under:

```text
r13946001/pathMNIST/
```

It includes:

- multi-dataset MedMNIST support;
- ResNet18 / small CNN / MobileNetV3-small model selection;
- IID and Dirichlet non-IID partitioning;
- event-driven async simulation with heterogeneous delays;
- CSV logging, summary JSON, plotting, checkpointing;
- multi-seed mean/std analysis;
- CAA-v2 ablation and distributed-system metrics.

Important reports:

```text
r13946001/REPORT_NOTES.md
r13946001/FINAL_RESULTS_SUMMARY.md
r13946001/NOVELTY_ASSESSMENT.md
r13946001/presentation/
```

## MLLM DEMO Track

The LLM/MLLM code and report support files are under:

```text
r13946001_MLLM/
```

Important files:

```text
r13946001_MLLM/README.md
r13946001_MLLM/FINAL_RESULTS_SUMMARY.md
r13946001_MLLM/EXPERIMENT_COMPLETENESS_AUDIT.md
r13946001_MLLM/presentation/LLM_MLLM_DEMO_REFERENCE.md
r13946001_MLLM/scripts/create_demo_final_deck.py
r13946001_MLLM/scripts/render_paired_example_images.py
```

## Repository Layout

```text
pathMNIST/                         Original team baseline code
llm/                               Teammate LLM prototype / baseline area
r13946001/                         Safe MedMNIST experiment copy
r13946001_MLLM/                    Safe LLM/MLLM experiment copy
PP-DMA_Demo_Final_Package/         Clean final DEMO handoff package
PP-DMA：Privacy-Preserving Distributed Model Adaptation.md
                                   Project brief / proposal notes
```

Generated outputs are ignored by git:

```text
results/
figures/
logs/
checkpoints/
data/
__pycache__/
*.egg-info/
```

## Quick Start: MedMNIST Smoke Test

```bash
cd r13946001/pathMNIST
python -m pip install -e .

PYTHONPATH=src python src/fed_pathmnist/run.py \
  --synthetic --method caa_fedbuff_v2 \
  --events 4 --clients 2 --buffer-size 2 \
  --model small_cnn --device cpu
```

## Quick Start: LLM/MLLM Smoke Test

```bash
cd r13946001_MLLM
python -m pip install -e .

PYTHONPATH=src python src/fed_mllm/run.py \
  --synthetic --task text_mcqa --dataset mmlu \
  --model tiny_text --method caa_fedbuff_v2 \
  --events 4 --clients 2 --buffer-size 2 --device cpu
```

Render the final DEMO deck from current compact summaries:

```bash
cd r13946001_MLLM
python scripts/create_demo_final_deck.py
```

## Reporting Boundaries

Use conservative wording:

- The project is privacy-preserving in the FL sense that raw data stays local in
  simulation.
- It does not implement cryptographic secure aggregation, differential privacy,
  or a real multi-hospital deployment.
- CAA-v2 is a deterministic, explainable course-project method, not a claimed
  SOTA publication.
- The LLM/MLLM results are demo and feasibility evidence; the Qwen3-VL PMC-VQA
  result is a pilot, while local Qwen2.5-VL is a small diagnostic matrix.

## License and Usage

This repository is shared as educational and research/demo material for the
Distributed Computing Systems course project.  It is not medical software and
must not be used for clinical diagnosis or treatment decisions.

Upstream datasets, pretrained models, and libraries remain governed by their own
licenses and terms.  Before any public release beyond the course context, the
team should add a formal `LICENSE` file and verify that all dataset/model usage
is compatible with the intended distribution.
