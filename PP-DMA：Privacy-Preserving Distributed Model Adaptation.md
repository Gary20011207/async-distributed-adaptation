# PP-DMA: Privacy-Preserving Distributed Model Adaptation

**Clockless Federated Adaptation for Medical AI**

This is the public-facing report for Group 4.  The project started as
**Asynchronous Distributed ML Adaptation** and evolved into a distributed
systems study of privacy-preserving medical model adaptation:

> When medical data must stay local, can asynchronous federated adaptation
> approach Sync FedAvg while controlling stale, conflicting, and fast-client
> dominated updates?

The main idea is simple: hospitals and devices do not train at the same speed.
Fast clients may send many updates, while slow clients may return updates
computed from old global models.  This turns federated learning into a system
problem about event ordering, staleness, stragglers, and fairness.

## Members

| Name | Student ID | Department |
| --- | --- | --- |
| Guan-Yu Chen | R13946001 | Data Science |
| Guang-Cheng Zhang | R14922172 | Computer Science |
| Yu-Jia Zhang | R14922140 | Computer Science |

## Motivation

Medical AI is a natural distributed learning scenario:

- raw medical data should stay inside each hospital or device;
- different sites have different GPUs, network delays, and workloads;
- data distributions can differ across hospitals;
- synchronous FL is stable but waits for slow clients;
- asynchronous FL avoids the barrier but may receive stale or conflicting
  updates.

Therefore, the project focuses on a core question:

> Can we keep the throughput advantage of asynchronous FL while staying close to
> the correctness and stability of Sync FedAvg?

## Distributed Systems Framing

We do not assume a synchronized physical clock.  Instead, the server uses
logical model versions:

```text
server_version       = current global model version
client_start_version = version received by the client before local training
staleness            = server_version - client_start_version
```

The system is asynchronous because the server does not wait for all clients
before moving forward.  Logical versions are used only to measure staleness;
they are not a round barrier.

This lets us study four system issues:

1. stale updates from slow clients;
2. instability from conflicting updates;
3. straggler effects under heterogeneous delay;
4. fast-client domination in the event stream.

## Project Scope

The project has two connected tracks.

### Track 1: MedMNIST Image Classification

This is the main algorithmic track.  We run federated medical image
classification with MedMNIST datasets and compare synchronous and asynchronous
aggregation methods.

Main code and reports:

```text
r13946001/pathMNIST/
r13946001/REPORT_NOTES.md
r13946001/FINAL_RESULTS_SUMMARY.md
r13946001/presentation/
```

### Track 2: LLM/MLLM Medical QA and VQA DEMO

This is the demo extension.  We apply the same clockless aggregation idea to
LoRA / QLoRA adapter updates for medical closed-ended QA and VQA.

Main code and handoff:

```text
r13946001_MLLM/
PP-DMA_Demo_Final_Package/
PP-DMA_Demo_Final_Package/slides/PP-DMA_Demo_Final.pptx
```

## Methods

| Method | Role | Idea |
| --- | --- | --- |
| Sync FedAvg | baseline | Wait for selected clients, then average. |
| Naive Async | baseline | Apply each arriving update immediately. |
| Staleness Async | baseline | Downweight updates by logical staleness. |
| FedBuff-lite | baseline | Aggregate a buffer of async updates. |
| CAA-FedBuff | proposed v1 | Add direction agreement, clipping, and adaptive alpha. |
| CAA-v2 | final method | Add server trajectory memory and client fairness credit. |

CAA-v2 is our final method because it is simple, explainable, and directly tied
to distributed-systems concerns.  It is not meant to be a black-box accuracy
trick; it is a server-side rule for deciding how much to trust each async
update.

## CAA-v2 in Plain English

For each arriving client update, the server asks:

1. **How stale is it?** Old updates receive less weight.
2. **Does it agree with recent update directions?** More agreement means more
   trust.
3. **Is the update too large?** Outlier deltas are clipped.
4. **Has this client contributed too often?** Fairness credit reduces long-term
   fast-client domination.

For each buffered update:

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

Then it normalizes the weights and applies a weighted delta update.  The rule
is clockless because it uses logical versions, model deltas, and contribution
counts rather than synchronized physical time.

## CAA-v2 for LLM/MLLM Adapters

For LLMs and MLLMs, aggregating the entire model is too expensive.  The demo
therefore freezes the base model and aggregates only LoRA / QLoRA adapter
deltas:

```text
delta_i = local_adapter_i - global_adapter_at_client_start
```

The same CAA-v2 logic is then applied to adapter deltas: staleness, direction
agreement, server trajectory memory, client fairness, and adaptive alpha.

This makes the method practical for closed-ended medical QA/VQA, where answers
are limited to formats such as `A/B/C/D` or `yes/no`.

## Existing vs Ours

| Component | Source | Role |
| --- | --- | --- |
| FedAvg, FedAsync, FedBuff | existing | Main FL baselines. |
| MedMNIST, Qwen, LoRA / QLoRA | existing | Benchmarks, models, and adapter tools. |
| Clockless event-driven simulator | ours | Simulates async arrivals and logical staleness. |
| Fair update-budget protocol | ours | Compares async events with sync client-update count. |
| CAA-v2 aggregation rule | ours | Agreement/fairness-aware clockless aggregation. |
| Report pipeline | ours | Tracks accuracy, staleness, stability, simulated time, and client imbalance. |

## Image Classification Experiments

The MedMNIST track uses:

```text
9 MedMNIST datasets
6 methods
3 seeds per dataset/method
ResNet18
IID partition
fair budget = async events = sync rounds * clients
```

Datasets:

```text
PathMNIST, PneumoniaMNIST, BloodMNIST, OrganAMNIST, OrganCMNIST,
DermaMNIST, OCTMNIST, BreastMNIST, TissueMNIST
```

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
- CAA-v2 beats the strongest classic baseline on `5/9` datasets.
- Staleness-only aggregation is stable but often too conservative.
- CAA-FedBuff v1 has the highest peak accuracy, but CAA-v2 is the cleaner final
  method because it is more stable and easier to explain.

Recommended claim:

> Under a fair update budget, CAA-v2 makes clockless asynchronous FL approach
> Sync FedAvg while reducing the instability of naive async and avoiding the
> over-conservatism of staleness-only aggregation.

### Sync vs Async: What Should We Conclude?

Sync FedAvg is still the easiest method to trust when the system can afford to
wait.  It is stable because every round has a clear barrier.

Async methods are valuable for distributed systems because the server does not
wait for every slow client.  This better matches hospitals with different
hardware, network delays, and workloads.  The cost is that async updates may be
stale or conflicting.

Our results support a balanced conclusion:

- naive async is useful but can be less stable;
- staleness-only async is safer but can be too conservative;
- CAA-v2 keeps the no-barrier benefit of async while using agreement and
  fairness signals to stay close to Sync FedAvg.

So the answer is not simply "Sync is better" or "Async is better."  Sync is the
stable reference point.  Async is the more realistic distributed-system design.
CAA-v2 is our attempt to make async closer to sync without reintroducing a full
round barrier.

## LLM/MLLM DEMO Results

### Qwen3-VL + PMC-VQA Pilot

| Method | Correct / Total | Accuracy |
| --- | ---: | ---: |
| Sync FedAvg | 47 / 100 | 47.0% |
| Naive Async | 46 / 100 | 46.0% |
| Staleness Async | 45 / 100 | 45.0% |
| FedBuff | 44 / 100 | 44.0% |
| CAA-v2 | 47 / 100 | 47.0% |

Conservative demo claim:

> In this Qwen3-VL + PMC-VQA pilot, CAA-v2 ties Sync FedAvg and is the strongest
> async-family method in the run.

### Local Support Results

- Text MCQA with Qwen1.5-0.5B LoRA: CAA-v2 has the highest mean best accuracy,
  while Sync remains slightly better in final stability.
- Real Qwen2.5-VL 3B 4-bit QLoRA: closed-answer valid rate is 100% in the small
  feasibility diagnostic.
- Paired VQA examples show cases where CAA-v2 is correct while Naive Async or
  Sync is incorrect.

Important boundary:

> We did not find a valid same-question example where the base model is wrong
> and CAA-v2 is correct.  The correct statement is that CAA-v2 can correct some
> errors made by other federated aggregation baselines in selected diagnostics.

## Evaluation Metrics

| Metric | Meaning |
| --- | --- |
| Best accuracy | Best model quality reached during training. |
| Final accuracy | Whether the method remains stable at the end. |
| Stability drop | `best_acc - final_acc`; how much the method oscillates. |
| Async-Sync gap | Accuracy difference between async and Sync FedAvg. |
| Average / p95 staleness | Logical delay under async arrivals. |
| Client contribution Gini | Whether fast clients dominate accepted updates. |
| Invalid answer rate | Whether an LLM/MLLM follows the closed-answer format. |

## Deliverables

- Clockless event-driven async FL simulator.
- Sync, Naive Async, Staleness Async, FedBuff-lite, CAA-FedBuff, and CAA-v2.
- MedMNIST multi-dataset and multi-seed experiment summaries.
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

## License and Usage

This repository is shared as educational and research/demo material for the
Distributed Computing Systems course project.  It is not medical software and
must not be used for clinical diagnosis or treatment decisions.

Upstream datasets, pretrained models, and libraries remain governed by their own
licenses and terms.  Before any public release beyond the course context, the
team should add a formal `LICENSE` file and verify that all dataset/model usage
is compatible with the intended distribution.

## Limitations

This project is privacy-preserving in the federated-learning sense that raw
client data stays local in simulation.  It does not yet implement cryptographic
secure aggregation, differential privacy, or a real hospital network
deployment.

The LLM/MLLM results are demo and feasibility evidence.  The Qwen3-VL PMC-VQA
result is a pilot, and the real Qwen2.5-VL runs are intentionally small.

## Final Takeaway

PP-DMA shows that asynchronous federated adaptation is not only an ML problem.
It is also a distributed-systems problem about time, ordering, delay, stale
information, and fairness.

CAA-v2 provides a simple clockless aggregation rule.  Across MedMNIST it
approaches Sync FedAvg under a fair update budget.  In the LLM/MLLM demo, the
same idea is applied to adapter deltas, and the Qwen3-VL pilot shows CAA-v2
matching Sync while outperforming other async baselines.
