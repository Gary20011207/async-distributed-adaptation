# PP-DMA: Privacy-Preserving Distributed Model Adaptation

**Clockless Federated Adaptation for Medical AI**

This document is the public-facing project report for Group 4.  The project
started as **Asynchronous Distributed ML Adaptation** and gradually became a
broader distributed-systems study:

> When medical data must stay local, can we adapt image models and LLM/MLLM
> models through federated learning without waiting for every client, while
> still controlling stale and conflicting updates?

The key idea is simple.  Hospitals, devices, or departments do not train at the
same speed.  A fast client may send many updates, while a slow client may send
an update computed from an old global model.  This is useful for a distributed
systems project because it turns machine learning into a system problem:
ordering events, handling stragglers, and deciding how much to trust stale
updates.

## Members

| Name | Student ID | Department |
| --- | --- | --- |
| Guan-Yu Chen | R13946001 | Data Science |
| Guang-Cheng Zhang | R14922172 | Computer Science |
| Yu-Jia Zhang | R14922140 | Computer Science |

## Motivation

Medical AI is a natural privacy-preserving distributed learning scenario.

- A hospital may not be allowed to send raw images or clinical questions to a
  central server.
- Different hospitals may have different GPUs, network quality, workloads, and
  patient distributions.
- A synchronous training round is easy to reason about, but the server must wait
  for slow clients.
- An asynchronous system is faster, but the server receives updates that may be
  stale, conflicting, or dominated by fast clients.

This project asks:

> Under a fair update budget, can asynchronous federated adaptation approach
> Sync FedAvg while reducing the instability caused by stale updates and
> stragglers?

## Distributed Systems Framing

The central distributed-systems concept is **no global clock**.

We do not assume that all clients share a synchronized physical timestamp.
Instead, the server tracks logical model versions:

```text
server_version       = current global model version
client_start_version = version received by the client before local training
staleness            = server_version - client_start_version
```

This lets us describe stale updates without relying on physical time.  The
system is asynchronous because the server does not wait for all clients before
moving forward.  It processes updates as they arrive, or after a small async
buffer is filled.

This gives the project four concrete system questions:

1. What happens when the server does not use a round barrier?
2. How much does staleness hurt convergence and stability?
3. Do stragglers change the behavior of asynchronous learning?
4. Do fast clients dominate the aggregation stream?

## Project Scope

The project has two connected tracks.

### Track 1: MedMNIST Image Classification

This is the main algorithmic research track.  We simulate federated medical
image classification using MedMNIST datasets and compare synchronous and
asynchronous aggregation methods.

Main code:

```text
r13946001/pathMNIST/
```

Main outputs:

```text
r13946001/REPORT_NOTES.md
r13946001/FINAL_RESULTS_SUMMARY.md
r13946001/NOVELTY_ASSESSMENT.md
r13946001/presentation/
```

### Track 2: LLM/MLLM Medical QA and VQA DEMO

This is the demo-oriented extension.  We apply the same clockless aggregation
idea to LoRA / QLoRA adapter updates for medical text QA and visual question
answering.

Main code and report support:

```text
r13946001_MLLM/
PP-DMA_Demo_Final_Package/
```

The final handoff package is:

```text
PP-DMA_Demo_Final_Package/slides/PP-DMA_Demo_Final.pptx
PP-DMA_Demo_Final_Package/docs/LLM_MLLM_DEMO_REFERENCE.md
PP-DMA_Demo_Final_Package/docs/demo_final_speaker_notes_zh.md
```

## Methods

We compare standard baselines with our CAA-family methods.

| Method | Role | Idea |
| --- | --- | --- |
| Sync FedAvg | baseline | Wait for all selected clients in each round, then average. |
| Naive Async | baseline | Apply each arriving update immediately. |
| Staleness Async | baseline | Downweight an update based only on logical staleness. |
| FedBuff-lite | baseline | Collect a small buffer of async updates before aggregation. |
| CAA-FedBuff | proposed v1 | Add direction agreement, delta clipping, and adaptive alpha. |
| CAA-v2 | final method | Add server trajectory memory and client fairness credit. |

CAA-v2 is the final method because it has the clearest distributed-systems
interpretation.  It is not just trying to maximize one lucky peak accuracy.  It
tries to keep async training close to Sync FedAvg while making the server more
careful about stale, conflicting, and over-represented client updates.

## CAA-v2 in Plain English

When a client sends an update, the server asks four questions:

1. **How old is this update?**  
   A stale update receives less weight.

2. **Does this update point in a similar direction as other accepted updates?**  
   An update that agrees with the buffer and recent server trajectory receives
   more trust.

3. **Is this update unusually large?**  
   Large deltas are clipped to reduce unstable jumps.

4. **Has this client already contributed too often?**  
   A fast client receives less extra advantage over time, reducing domination.

For each buffered client update:

```text
delta_i = client_model_i - model_at_client_start_i
tau_i   = server_version - client_start_version_i
```

The server computes:

```text
raw_weight_i =
    num_examples_i
  * staleness_decay(tau_i)
  * agreement_factor_i
  * fairness_credit_i
```

Then the server normalizes these weights and applies a weighted delta update.
The rule is **clockless** because it uses logical versions, model deltas, and
client contribution counts instead of synchronized physical time.

## CAA-v2 for LLM/MLLM Adapters

For large language and multimodal models, aggregating the whole model is too
expensive.  Therefore, the LLM/MLLM demo uses adapter-level federated
adaptation.

The base model is frozen, and each client trains only a LoRA or QLoRA adapter:

```text
delta_i = local_adapter_i - global_adapter_at_client_start
```

The server aggregates adapter deltas using the same CAA-v2 logic:

- logical staleness;
- direction agreement;
- server trajectory memory;
- client fairness credit;
- adaptive alpha.

This makes the method practical enough for a demo with medical closed-ended
question answering and VQA.  The answer space is finite, such as `A/B/C/D` or
`yes/no`, so the evaluation can use accuracy and invalid-answer rate.

## What Is Existing vs What Is Ours

| Component | Source | Role |
| --- | --- | --- |
| FedAvg | existing | Synchronous FL baseline. |
| FedAsync / staleness-aware aggregation | existing | Async and stale-update baselines. |
| FedBuff | existing | Buffered async aggregation baseline. |
| MedMNIST | existing benchmark | Biomedical image classification datasets. |
| Qwen / Qwen2.5-VL / Qwen3-VL | existing models | Base LLM/MLLM backbones. |
| LoRA / QLoRA | existing adaptation method | Parameter-efficient local adaptation. |
| CAA-v2 aggregation rule | ours | Agreement/fairness-aware clockless aggregation. |
| Event-driven simulator | ours | Simulates async arrivals and logical staleness. |
| Fair update-budget protocol | ours | Compares async events with sync client-update count. |
| Report and visualization pipeline | ours | Tracks accuracy, staleness, stability, simulated time, and client imbalance. |

## Experimental Setup: Image Classification

The MedMNIST research track uses a fair multi-seed matrix:

```text
datasets = 9 MedMNIST datasets
methods  = 6 methods
seeds    = 42, 43, 44
model    = ResNet18
partition = IID
fair budget = async events = sync rounds * clients
```

Datasets:

```text
PathMNIST, PneumoniaMNIST, BloodMNIST, OrganAMNIST, OrganCMNIST,
DermaMNIST, OCTMNIST, BreastMNIST, TissueMNIST
```

Fairness controls:

| Control | Setting |
| --- | --- |
| Clients | 10 |
| Local epochs | 1 |
| Batch size | 128 |
| Learning rate | 0.01 with cosine scheduler |
| Async delay | Same heterogeneous setting across async baselines |
| Update budget | `async events = sync rounds * clients` |
| Seeds | 42, 43, 44 |

## Image Classification Results

Overall mean across the 9-dataset fair matrix:

| Method | Best Acc Mean | Final Acc Mean | Stability Drop |
| --- | ---: | ---: | ---: |
| Sync FedAvg | 0.7142 | 0.7121 | 0.0020 |
| Naive Async | 0.7132 | 0.7096 | 0.0036 |
| Staleness Async | 0.6770 | 0.6752 | 0.0017 |
| FedBuff-lite | 0.7090 | 0.7062 | 0.0028 |
| CAA-FedBuff | 0.7206 | 0.7158 | 0.0048 |
| CAA-v2 | 0.7169 | 0.7140 | 0.0029 |

Main findings:

- CAA-v2 beats Sync FedAvg in mean best accuracy on `6/9` datasets.
- CAA-v2 beats Sync FedAvg in mean final accuracy on `6/9` datasets.
- CAA-v2 beats the strongest classic baseline among Sync, Naive Async,
  Staleness Async, and FedBuff-lite on `5/9` datasets.
- Staleness-only aggregation is stable but often too conservative.
- CAA-FedBuff v1 has the highest peak accuracy, but a larger stability drop.
- CAA-v2 is the cleaner final method because it balances performance,
  stability, and interpretability.

Recommended image-classification claim:

> Under a fair update budget, CAA-v2 makes clockless asynchronous FL approach
> Sync FedAvg across diverse MedMNIST datasets, while reducing the instability
> of naive async and avoiding the over-conservatism of staleness-only
> aggregation.

## LLM/MLLM DEMO Results

The final demo integrates peer Qwen3-VL results and local LLM/MLLM support
experiments.

### Main VQA Demo: Qwen3-VL + PMC-VQA

| Method | Correct / Total | Accuracy |
| --- | ---: | ---: |
| Sync FedAvg | 47 / 100 | 47.0% |
| Naive Async | 46 / 100 | 46.0% |
| Staleness Async | 45 / 100 | 45.0% |
| FedBuff | 44 / 100 | 44.0% |
| CAA-v2 | 47 / 100 | 47.0% |

Conservative demo claim:

> In the Qwen3-VL + PMC-VQA pilot, CAA-v2 ties Sync FedAvg and is the strongest
> async-family method in that run.

### Local Text MCQA Support

Local text QA experiments use Qwen1.5-0.5B with LoRA on closed-ended medical
QA datasets.  CAA-v2 has the highest mean best accuracy, while Sync FedAvg
remains slightly better in final stability.  This supports the method but does
not claim a universal win.

### Local Real Qwen2.5-VL Feasibility

The real Qwen2.5-VL 3B 4-bit QLoRA diagnostic shows a 100% closed-answer valid
rate.  This means adapter aggregation did not break the output format, but the
run is intentionally small and should be treated as feasibility evidence rather
than a large benchmark.

### Paired VQA Examples

The demo package includes four same-question VQA examples with images:

```text
VQA-RAD #16
PathVQA #40
PathVQA #50
VQA-RAD #31
```

These examples compare Base, Naive Async, Sync, and CAA-v2 on the same image
and question.  They show cases where CAA-v2 is correct while Naive Async or
Sync is incorrect.

Important boundary:

> We did not find a valid same-question example where the base model is wrong
> and CAA-v2 is correct.  Therefore, the demo should not claim "before training
> wrong, after training correct" for the same question.  The correct statement
> is that CAA-v2 can correct some errors made by other federated aggregation
> baselines in selected same-question diagnostics.

## Evaluation Metrics

We evaluate both ML performance and system behavior.

| Metric | Meaning |
| --- | --- |
| Best accuracy | Best model quality reached during training. |
| Final accuracy | Whether the method remains stable at the end. |
| Stability drop | `best_acc - final_acc`; how much the method oscillates. |
| Async-Sync best gap | Peak accuracy cost of removing the sync barrier. |
| Async-Sync final gap | Whether async converges close to Sync FedAvg. |
| Average / p95 staleness | Logical delay under async arrivals. |
| Client contribution Gini | Whether fast clients dominate accepted updates. |
| Invalid answer rate | Whether an LLM/MLLM follows the closed-answer format. |

## Deliverables

Implemented and prepared:

- Clockless event-driven async FL simulator.
- Sync, Naive Async, Staleness Async, FedBuff-lite, CAA-FedBuff, and CAA-v2.
- MedMNIST multi-dataset experiments with multi-seed summaries.
- Non-IID and straggler/delay stress analysis.
- LLM/MLLM adapter-level FL prototype.
- Qwen3-VL, Qwen2.5-VL, and text MCQA result summaries.
- Final demo slides and a clean handoff package.

Important handoff files:

```text
PP-DMA_Demo_Final_Package/slides/PP-DMA_Demo_Final.pptx
PP-DMA_Demo_Final_Package/docs/LLM_MLLM_DEMO_REFERENCE.md
PP-DMA_Demo_Final_Package/assets/paired_image_examples.png
r13946001/REPORT_NOTES.md
r13946001_MLLM/FINAL_RESULTS_SUMMARY.md
```

## How To Run

MedMNIST smoke test:

```bash
cd r13946001/pathMNIST
python -m pip install -e .

PYTHONPATH=src python src/fed_pathmnist/run.py \
  --synthetic --method caa_fedbuff_v2 \
  --events 4 --clients 2 --buffer-size 2 \
  --model small_cnn --device cpu
```

LLM/MLLM smoke test:

```bash
cd r13946001_MLLM
python -m pip install -e .

PYTHONPATH=src python src/fed_mllm/run.py \
  --synthetic --task text_mcqa --dataset mmlu \
  --model tiny_text --method caa_fedbuff_v2 \
  --events 4 --clients 2 --buffer-size 2 --device cpu
```

Regenerate the final demo deck:

```bash
cd r13946001_MLLM
python scripts/create_demo_final_deck.py
```

## Limitations

This project is privacy-preserving in the federated-learning sense that raw
client data stays local in the simulation.  It does not yet implement
cryptographic secure aggregation, differential privacy, or a real hospital
network deployment.

The LLM/MLLM experiments are demo and feasibility evidence.  The Qwen3-VL
PMC-VQA result is a pilot, and the real Qwen2.5-VL runs are intentionally small.
Future work should include larger VQA evaluation, stronger paired before/after
diagnostics, real network traces, secure aggregation, and privacy guarantees.

## Final Takeaway

PP-DMA shows that asynchronous federated adaptation is not only an ML problem.
It is a distributed-systems problem about time, ordering, delay, stale
information, and fairness.

CAA-v2 provides a simple and explainable clockless aggregation rule.  Across
the MedMNIST experiments it approaches Sync FedAvg under a fair update budget.
In the LLM/MLLM demo, the same idea can be applied to adapter deltas, and the
Qwen3-VL pilot shows CAA-v2 matching Sync while outperforming other async
baselines.
