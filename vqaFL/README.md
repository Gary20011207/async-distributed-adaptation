# VQA-FL: Qwen3-VL LoRA Federated Learning Plan

This folder is the next project extension: testing whether CAA-style asynchronous aggregation transfers from CNN image classification to closed-ended medical VQA.

## Scope

```text
Base model: Qwen/Qwen3-VL-2B-Instruct
Training: LoRA or QLoRA only
Task: closed-ended / multiple-choice medical VQA
Primary dataset: PMC-VQA subset
First metric: option accuracy
First clients: 3, then 5
```

The goal is not to fully fine-tune a VLM. Each client trains adapter weights, and the server aggregates adapter deltas.

## Why Qwen3-VL-2B

- It is the smallest official Qwen3-VL instruct model.
- It is a vision-language model, so it can handle image + question prompts.
- It is lighter than Qwen2.5-VL-3B and more realistic for a first FL proof of concept.
- Larger fallback: `Qwen/Qwen3-VL-4B-Instruct`.

## Methods

Planned baselines:

```text
Centralized LoRA
Sync FedAvg-LoRA
Naive Async-LoRA
Staleness Async-LoRA
FedBuff-LoRA
CAA-v2-LoRA
```

CAA-v2-LoRA maps the existing CAA idea to adapter deltas:

```text
delta_i = local_lora_i_after_train - lora_global_at_client_start
raw_weight_i =
    num_examples_i
  * staleness_decay(tau_i)
  * agreement_factor_i
  * fairness_credit_i
```

Agreement is computed as cosine similarity between LoRA deltas, not full model weights.

## Data Format

The first loader target is a JSONL multiple-choice format:

```json
{"image": "path/to/image.png", "question": "...", "choices": ["A", "B", "C", "D"], "answer": "B", "client_id": 0}
```

The prompt should force a short answer:

```text
Answer with one letter only: A, B, C, or D.
```

This keeps evaluation deterministic and avoids open-ended answer normalization.

## Runtime Notes

The project venv now has the first VQA dependencies installed:

```text
transformers 5.9.0
peft 0.19.1
datasets 4.8.5
accelerate 1.13.0
qwen-vl-utils 0.0.14
```

Do not run CPU training. Use CUDA for any real training command. The local sandbox cannot see CUDA, so GPU checks must run outside the sandbox.

The first Qwen3-VL-2B model smoke test succeeded with rank 4 LoRA:

```text
Total parameters: 2,131,890,176
Trainable LoRA parameters: 4,358,144
Trainable ratio: 0.2044%
Adapter tensors: 392
```

The Hugging Face cache is about 15 GB after downloading the model files.

A tiny local PMC-VQA subset was exported under `vqaFL/data/pmc_vqa_tiny/` for smoke tests. It is ignored by git.

## Smoke Checks

Adapter-level CAA logic can be checked without downloading Qwen:

```bash
cd vqaFL
PYTHONPATH=src python -m vqa_fl.run --mode adapter-smoke --device cuda
```

This only checks adapter-delta aggregation on synthetic LoRA tensors. It is not a VQA training run.

Synthetic data partition check:

```bash
cd vqaFL
PYTHONPATH=src python -m vqa_fl.run --mode data-smoke --clients 3 --max-examples 12
```

Qwen3-VL-2B + LoRA load check:

```bash
cd vqaFL
python -m vqa_fl.run --mode model-smoke --model Qwen/Qwen3-VL-2B-Instruct --lora-rank 4 --lora-alpha 8 --dtype bf16
```

PMC-VQA streaming smoke test:

```bash
cd vqaFL
timeout 30 python -m vqa_fl.run --mode data-smoke --data-source hf-pmc-vqa --clients 2 --max-examples 2
```

The timeout is intentional for now. Hugging Face streaming returns the sample correctly, but the Python process can remain alive during shutdown in this environment.

Export a tiny local PMC-VQA subset:

```bash
cd vqaFL
timeout 60 python -m vqa_fl.run --mode export-data --data-source hf-pmc-vqa --max-examples 6 --clients 3 --output-jsonl data/pmc_vqa_tiny/examples.jsonl --output-dir data/pmc_vqa_tiny
```

One-client local LoRA training smoke:

```bash
cd vqaFL
python -m vqa_fl.run --mode local-train-smoke --data-source jsonl --data-jsonl data/pmc_vqa_tiny/examples.jsonl --max-examples 4 --model Qwen/Qwen3-VL-2B-Instruct --lora-rank 4 --lora-alpha 8 --dtype bf16 --batch-size 1 --grad-accum 2 --local-epochs 1
```

Latest tiny run:

```text
train_examples: 3
eval_examples: 1
train_steps: 3
train_loss: about 0.61
eval_accuracy: 0.0 on 1 sample
```

The tiny accuracy is not meaningful. It only confirms that loading, visual prompt processing, backward pass, adapter extraction, generation, and closed-ended parsing all execute.

## Tiny FL Run

The first real 3-client Qwen3-VL LoRA FL run has been executed on the tiny local PMC-VQA subset:

```bash
cd vqaFL
python -m vqa_fl.run --mode fl-run --method all --staleness-decay inverse --data-source jsonl --data-jsonl data/pmc_vqa_tiny/examples.jsonl --max-examples 6 --clients 3 --rounds 3 --buffer-size 3 --server-alpha 0.5 --model Qwen/Qwen3-VL-2B-Instruct --lora-rank 4 --lora-alpha 8 --dtype bf16 --batch-size 1 --grad-accum 2 --local-epochs 1 --eval-every 3 --results-json results/fl_all_tiny_3round_inverse.json
```

Result summary:

```text
Dataset: 6 examples total, 4 train / 2 eval after split
Clients: 3 IID clients, train sizes [2, 1, 1]
Rounds: 3
Methods: Sync FedAvg, Naive Async, Staleness Async, FedBuff, CAA-v2
Final option accuracy: 0.0 on 2 eval samples for all methods
```

Important diagnostics:

```text
Sync FedAvg final server version: 3
Naive Async final server version: 9
Staleness Async final server version: 9
FedBuff final server version: 3
CAA-v2 final server version: 3
Staleness Async inverse alphas in round 1: 0.5, 0.25, 0.1667
CAA-v2 adaptive alphas over rounds: 0.5521, 0.5645, 0.5666
CAA-v2 mean agreement over rounds: 0.5559, 0.5848, 0.5820
```

This run validates the end-to-end FL control path. The dataset is too small to interpret accuracy.

## Performance Run 60

First performance-oriented run:

```bash
cd vqaFL
python -m vqa_fl.run --mode fl-run --method all --staleness-decay inverse --data-source jsonl --data-jsonl data/pmc_vqa_perf_60/examples.jsonl --max-examples 60 --clients 3 --rounds 3 --buffer-size 3 --server-alpha 0.5 --model Qwen/Qwen3-VL-2B-Instruct --lora-rank 4 --lora-alpha 8 --dtype bf16 --batch-size 1 --grad-accum 4 --local-epochs 1 --eval-fraction 0.2 --eval-every 3 --results-json results/fl_all_perf60_3round_inverse.json
```

Setup:

```text
Dataset: PMC-VQA local subset, 60 examples
Split: 48 train / 12 eval
Clients: 3 IID clients, train sizes [16, 16, 16]
Rounds: 3
LoRA: rank 4, alpha 8
Staleness decay: inverse
```

Result summary:

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

This is the first usable performance measurement, but the eval set still has only 12 examples. Treat it as a pilot result.

## Performance Run 200 / Eval 100

Larger eval run:

```bash
cd vqaFL
python -m vqa_fl.run --mode fl-run --method all --staleness-decay inverse --data-source jsonl --data-jsonl data/pmc_vqa_perf_200/examples.jsonl --max-examples 200 --clients 3 --rounds 3 --buffer-size 3 --server-alpha 0.5 --model Qwen/Qwen3-VL-2B-Instruct --lora-rank 4 --lora-alpha 8 --dtype bf16 --batch-size 1 --grad-accum 4 --local-epochs 1 --eval-fraction 0.5 --eval-every 0 --results-json results/fl_all_perf200_eval100_3round_inverse.json
```

Setup:

```text
Dataset: PMC-VQA local subset, 200 examples
Split: 100 train / 100 eval
Clients: 3 IID clients, train sizes [34, 33, 33]
Rounds: 3
LoRA: rank 4, alpha 8
Staleness decay: inverse
```

Result summary:

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

```text
Sync FedAvg and CAA-v2 tie at 47/100.
Naive Async is close at 46/100.
Staleness Async and FedBuff trail slightly at 45/100 and 44/100.
```

This is still a small single-seed result, but it is now measured on 100 eval examples.
