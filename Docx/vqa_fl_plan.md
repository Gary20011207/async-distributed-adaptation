# VQA-FL Next Step Plan

## Goal

Test whether CAA-style aggregation transfers from CNN medical image classification to closed-ended medical VQA with LoRA adapters.

## First Configuration

```text
Model: Qwen/Qwen3-VL-2B-Instruct
Task: closed-ended / multiple-choice medical VQA
Training: LoRA or QLoRA
Aggregation unit: LoRA adapter delta
Metric: option accuracy
Clients: 3 first, then 5
Dataset: PMC-VQA subset first
```

## Why Closed-Ended VQA

- Accuracy is simple and deterministic.
- The model can be forced to answer `A`, `B`, `C`, or `D`.
- It avoids open-ended answer normalization.
- It should train and evaluate faster than generative free-form VQA.

## Method Mapping

CAA on CNNs used full model deltas. CAA on Qwen3-VL uses LoRA deltas:

```text
delta_i = local_lora_i_after_train - global_lora_at_client_start
```

Then CAA-v2-LoRA applies:

```text
raw_weight_i =
    num_examples_i
  * staleness_decay(tau_i)
  * agreement_factor_i
  * fairness_credit_i
```

Where:

- `tau_i` is logical staleness.
- `agreement_factor_i` is cosine agreement between LoRA deltas.
- `fairness_credit_i` reduces fast-client domination.
- server trajectory memory is an EMA of accepted LoRA deltas.

## Planned Baselines

```text
Centralized LoRA
Sync FedAvg-LoRA
Naive Async-LoRA
Staleness Async-LoRA
FedBuff-LoRA
CAA-v2-LoRA
```

## Current Scaffold

New folder:

```text
vqaFL/
```

Included now:

- Qwen3-VL-2B project README.
- JSONL and Hugging Face PMC-VQA closed-ended VQA loader.
- IID and `client_id` client partition helpers.
- Option accuracy parser.
- Adapter-level FedBuff and CAA-v2 aggregation logic.
- LoRA server orchestration layer for sync/async methods.
- Qwen3-VL LoRA loader and adapter state helpers.
- Local LoRA trainer and closed-ended evaluator interface.
- CUDA adapter smoke mode that does not require downloading Qwen.
- Qwen3-VL model smoke mode.

## Current Verification

Environment:

```text
GPU: NVIDIA GeForce RTX 5090, 32 GB VRAM
torch: 2.11.0+cu130
transformers: 5.9.0
peft: 0.19.1
datasets: 4.8.5
accelerate: 1.13.0
```

Verified:

- `python -m compileall vqaFL/src` passes.
- Synthetic data smoke gives 3 clients with 4 samples each.
- PMC-VQA HF streaming can read samples from `OctoMed/PMC-VQA`.
- A tiny local PMC-VQA subset was exported to `vqaFL/data/pmc_vqa_tiny/`.
- Prompt option cleanup works, avoiding duplicate `A. A:` labels.
- CUDA adapter smoke passes.
- Qwen3-VL-2B model smoke passes with rank 4 LoRA.
- One-client local Qwen3-VL LoRA train smoke passes on 4 examples.
- 3-client Qwen3-VL LoRA FL run passes for all planned methods on the tiny subset.

Qwen3-VL-2B LoRA smoke result:

```text
Total parameters: 2,131,890,176
Trainable parameters: 4,358,144
Trainable ratio: 0.2044%
Adapter tensors: 392
```

Known environment note:

- The sandbox cannot access CUDA, so GPU commands need to run outside the sandbox.
- Hugging Face streaming returns PMC-VQA samples correctly, but the process can hang during shutdown; use `timeout 30` for data smoke until we replace streaming with a cached subset file.
- Hugging Face cache is currently about 15 GB after model download.

Local train smoke result:

```text
Input subset: 4 examples
Train examples: 3
Eval examples: 1
LoRA rank: 4
Train steps: 3
Train loss: about 0.61
Eval accuracy: 0.0 on 1 sample
```

This accuracy is only a pipeline smoke result. It is not an experiment result.

## Tiny FL Run Result

Command target:

```text
mode: fl-run
method: all
staleness_decay: inverse
model: Qwen/Qwen3-VL-2B-Instruct
LoRA rank/alpha: 4 / 8
clients: 3
rounds: 3
local epochs: 1
batch size: 1
grad accumulation: 2
subset: vqaFL/data/pmc_vqa_tiny/examples.jsonl
result file: vqaFL/results/fl_all_tiny_3round_inverse.json
```

Data split:

```text
Total examples: 6
Train examples: 4
Eval examples: 2
Client train sizes: [2, 1, 1]
```

Summary:

| Method | Final server version | Final option accuracy |
| --- | ---: | ---: |
| Sync FedAvg | 3 | 0.0 / 2 |
| Naive Async | 9 | 0.0 / 2 |
| Staleness Async | 9 | 0.0 / 2 |
| FedBuff | 3 | 0.0 / 2 |
| CAA-v2 | 3 | 0.0 / 2 |

Diagnostics:

```text
Staleness Async inverse alphas in round 1: 0.5, 0.25, 0.1667
CAA-v2 adaptive alphas over rounds: 0.5521, 0.5645, 0.5666
CAA-v2 mean agreement over rounds: 0.5559, 0.5848, 0.5820
```

Interpretation:

- This is an end-to-end pipeline validation, not a performance claim.
- The tiny subset is too small for accuracy to be meaningful.
- The next useful step is scaling the cached subset size before comparing methods.

## Performance Run 60

The first performance-oriented run used a 60-example cached PMC-VQA subset.

Command target:

```text
mode: fl-run
method: all
staleness_decay: inverse
model: Qwen/Qwen3-VL-2B-Instruct
LoRA rank/alpha: 4 / 8
clients: 3
rounds: 3
local epochs: 1
batch size: 1
grad accumulation: 4
eval fraction: 0.2
subset: vqaFL/data/pmc_vqa_perf_60/examples.jsonl
result file: vqaFL/results/fl_all_perf60_3round_inverse.json
```

Data split:

```text
Total examples: 60
Train examples: 48
Eval examples: 12
Client train sizes: [16, 16, 16]
Train answer histogram: A=8, B=13, C=18, D=9
```

Performance summary:

| Method | Correct / Eval | Accuracy | Final server version | Round 3 mean train loss |
| --- | ---: | ---: | ---: | ---: |
| Sync FedAvg | 6 / 12 | 0.5000 | 3 | 0.4225 |
| Naive Async | 6 / 12 | 0.5000 | 9 | 0.3700 |
| Staleness Async | 5 / 12 | 0.4167 | 9 | 0.3968 |
| FedBuff | 5 / 12 | 0.4167 | 3 | 0.4783 |
| CAA-v2 | 6 / 12 | 0.5000 | 3 | 0.4483 |

Diagnostics:

```text
Staleness Async inverse alphas in round 1: 0.5, 0.25, 0.1667
CAA-v2 adaptive alphas over rounds: 0.5748, 0.5938, 0.5805
CAA-v2 mean agreement over rounds: 0.7976, 0.7957, 0.7068
CAA-v2 server agreement over rounds: 0.0000, 0.6145, 0.4543
```

Interpretation:

- This is a pilot performance measurement, not a final result.
- Sync FedAvg, Naive Async, and CAA-v2 tie at 6/12 on this small eval split.
- Staleness Async and FedBuff are 5/12 on this split.
- The eval set is still too small; next run should use at least 100-200 examples or repeated seeds.

## Performance Run 200 / Eval 100

The next run used 200 total examples and measured final accuracy on 100 eval examples.

Command target:

```text
mode: fl-run
method: all
staleness_decay: inverse
model: Qwen/Qwen3-VL-2B-Instruct
LoRA rank/alpha: 4 / 8
clients: 3
rounds: 3
local epochs: 1
batch size: 1
grad accumulation: 4
eval fraction: 0.5
eval every: 0, final eval only
subset: vqaFL/data/pmc_vqa_perf_200/examples.jsonl
result file: vqaFL/results/fl_all_perf200_eval100_3round_inverse.json
```

Data split:

```text
Total examples: 200
Train examples: 100
Eval examples: 100
Client train sizes: [34, 33, 33]
Train answer histogram: A=23, B=26, C=32, D=19
```

Performance summary:

| Method | Correct / Eval | Accuracy | Final server version | Round 3 mean train loss |
| --- | ---: | ---: | ---: | ---: |
| Sync FedAvg | 47 / 100 | 0.4700 | 3 | 0.3316 |
| Naive Async | 46 / 100 | 0.4600 | 9 | 0.2447 |
| Staleness Async | 45 / 100 | 0.4500 | 9 | 0.3676 |
| FedBuff | 44 / 100 | 0.4400 | 3 | 0.3804 |
| CAA-v2 | 47 / 100 | 0.4700 | 3 | 0.3648 |

Diagnostics:

```text
Staleness Async inverse alphas in round 1: 0.5, 0.25, 0.1667
CAA-v2 adaptive alphas over rounds: 0.5652, 0.5657, 0.5581
CAA-v2 mean agreement over rounds: 0.6957, 0.5899, 0.5695
CAA-v2 server agreement over rounds: 0.0000, 0.3315, 0.1510
```

Interpretation:

- Sync FedAvg and CAA-v2 tie at 47/100.
- Naive Async is close at 46/100.
- Staleness Async and FedBuff trail slightly at 45/100 and 44/100.
- This is still single-seed, but eval size is now large enough for a first comparison table.

Next implementation steps:

1. Run repeated seeds for 3-client method comparison.
2. Add prediction logging to inspect wrong answers.
3. Tune prompt/answer parsing if generated answers are not stable A-D outputs.
4. Try 5-client partition after repeated-seed 3-client results are stable.
