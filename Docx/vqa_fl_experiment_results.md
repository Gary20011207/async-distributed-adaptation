# Qwen3-VL VQA Federated Learning Experiment Results

## Summary

This document records the current VQA federated learning experiment results. The goal is to test whether the asynchronous aggregation ideas from the PathMNIST experiments can transfer to a closed-ended VQA task with LoRA adapter training.

Main takeaway:

- CAA-v2 ties Sync FedAvg on the 100-example eval run: `47/100`.
- Naive Async is close: `46/100`.
- Staleness Async and FedBuff are slightly lower on the same split: `45/100` and `44/100`.
- These are single-seed pilot results, not final statistically stable results.

## Model And Training Setup

```text
Base model: Qwen/Qwen3-VL-2B-Instruct
Training method: LoRA adapter tuning
Full fine-tuning: no
LoRA rank: 4
LoRA alpha: 8
dtype: bf16
Batch size: 1
Gradient accumulation: 4
Local epochs per client per round: 1
Clients: 3
Partition: IID
Task type: closed-ended / multiple-choice VQA
Metric: option accuracy
```

The prompt forces the model to answer one option letter:

```text
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer with one letter only: A, B, C, or D.
```

The evaluator parses generated output as `A`, `B`, `C`, or `D`.

## Dataset

Dataset source:

```text
OctoMed/PMC-VQA
```

The dataset is exported to local JSONL plus image files before training, to avoid depending on Hugging Face streaming during training.

Main run subset:

```text
Local file: vqaFL/data/pmc_vqa_perf_200/examples.jsonl
Total examples: 200
Train examples: 100
Eval examples: 100
Client train sizes: [34, 33, 33]
Train answer histogram: A=23, B=26, C=32, D=19
```

The local `data/` and `results/` folders are intentionally ignored by git. The experiment numbers below are copied from the local JSON result files.

## Compared Methods

```text
sync_fedavg
naive_async
staleness_async
fedbuff
caa_v2
```

Method notes:

- `sync_fedavg`: all clients train from the same global adapter state, then the server averages all LoRA deltas.
- `naive_async`: each client update is applied immediately with fixed server alpha.
- `staleness_async`: each client update is applied immediately, but stale updates use a smaller effective alpha.
- `fedbuff`: client updates are buffered and then applied as a batch.
- `caa_v2`: buffered async aggregation with agreement weighting, fairness weighting, adaptive alpha, and server trajectory memory.

## Main Result: 200 Examples / 100 Eval

Command:

```bash
cd vqaFL
python -m vqa_fl.run \
  --mode fl-run \
  --method all \
  --staleness-decay inverse \
  --data-source jsonl \
  --data-jsonl data/pmc_vqa_perf_200/examples.jsonl \
  --max-examples 200 \
  --clients 3 \
  --rounds 3 \
  --buffer-size 3 \
  --server-alpha 0.5 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --lora-rank 4 \
  --lora-alpha 8 \
  --dtype bf16 \
  --batch-size 1 \
  --grad-accum 4 \
  --local-epochs 1 \
  --eval-fraction 0.5 \
  --eval-every 0 \
  --results-json results/fl_all_perf200_eval100_3round_inverse.json
```

Result table:

| Method | Correct / Eval | Accuracy | Final Server Version | Round 3 Mean Train Loss |
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

- Sync FedAvg and CAA-v2 are tied at `47/100`.
- Naive Async is close at `46/100`.
- Staleness Async and FedBuff trail slightly in this split.
- CAA-v2 is not clearly better yet, but it reaches Sync FedAvg performance while staying in the async aggregation family.

## Pilot Result: 60 Examples / 12 Eval

This earlier run used 60 total examples with `48 train / 12 eval`.

| Method | Correct / Eval | Accuracy | Final Server Version | Round 3 Mean Train Loss |
| --- | ---: | ---: | ---: | ---: |
| Sync FedAvg | 6 / 12 | 0.5000 | 3 | 0.4225 |
| Naive Async | 6 / 12 | 0.5000 | 9 | 0.3700 |
| Staleness Async | 5 / 12 | 0.4167 | 9 | 0.3968 |
| FedBuff | 5 / 12 | 0.4167 | 3 | 0.4783 |
| CAA-v2 | 6 / 12 | 0.5000 | 3 | 0.4483 |

This pilot result is directionally consistent with the 100-eval run, but the eval size is too small for strong claims.

## Smoke Result: 6 Examples / 2 Eval

The first end-to-end run used only 6 examples and was treated as pipeline validation.

```text
Train examples: 4
Eval examples: 2
Methods: all five methods
Accuracy: 0/2 for all methods
```

This run validated that model loading, LoRA training, adapter extraction, server aggregation, and VQA evaluation all execute.

## Limitations

- Current results are single-seed only.
- The main eval set has 100 examples, which is useful for a first table but still small.
- Only IID partitioning has been tested for VQA-FL so far.
- Only 3 clients have been used in the VQA-FL runs.
- Local training uses `local_epochs=1` and only 3 global rounds.
- Prediction logs are not yet saved, so wrong-answer analysis is not available.
- The evaluator only scores parsed option letters; answer formatting errors may still affect accuracy.

## Next Experiments

Recommended next steps:

1. Run repeated seeds for the 200-example / 100-eval setup.
2. Add prediction logging for generated text, parsed option, and ground truth.
3. Test 5 clients with the same 100-eval setup.
4. Increase rounds from 3 to 5 or 10 before increasing local epochs.
5. Add a non-IID client partition for VQA.

## Current Status

The current VQA-FL implementation is ready for repeated-seed performance testing. The best current result is a tie between Sync FedAvg and CAA-v2 at `47/100` on the 100-example eval split.
