# PP-DMA：Privacy-Preserving Distributed Model Adaptation

**Clockless Federated Adaptation for Medical Imaging**

This document is the updated project brief for Group 4. The project started as
**Asynchronous Distributed ML Adaptation** and has converged into a concrete
federated learning system study:

> In a privacy-preserving medical FL setting where raw data stays local, how
> should a server aggregate asynchronous updates when there is no global clock
> and some updates are stale or conflicting?

## Members

| Name | Student ID | Department |
|---|---|---|
| 陳冠宇 | R13946001 | Data Science |
| 張光澄 | R14922172 | Computer Science |
| 張育嘉 | R14922140 | Computer Science |

## Motivation

Medical AI is a good example of distributed learning:

- Hospitals should not centralize raw medical images.
- Different hospitals have different GPUs, network conditions, and workloads.
- Patient populations and imaging devices differ, so data can be non-IID.
- A slow hospital may return an update trained from an old global model.

Synchronous FL is stable but waits for slow hospitals. Asynchronous FL avoids the
barrier, but stale updates can hurt convergence. Therefore, the project asks:

> Can asynchronous FL approach Sync FedAvg under a fair update budget while
> controlling stale updates, stragglers, and fast-client domination?

## Distributed Systems Framing

The key concept is **no global clock**. We do not assume synchronized physical
time across clients. Instead, the server assigns logical model versions:

```text
server_version = current global model version
client_start_version = version received by a client before local training
staleness = server_version - client_start_version
```

This turns asynchronous FL into an event-ordering problem:

- The server observes update arrival order.
- Arrival order is not the same as training start order.
- Late updates may contain stale or conflicting information.
- Fast clients can dominate the event stream if aggregation is naive.

## Final Direction

We focus on **federated medical image classification** with MedMNIST and
ResNet18. The implementation is a reproducible course-project research/demo
codebase, not a production FL framework.

Main runnable code:

```text
r13946001/pathMNIST/
```

Main report files:

```text
r13946001/REPORT_NOTES.md
r13946001/FINAL_RESULTS_SUMMARY.md
r13946001/NOVELTY_ASSESSMENT.md
r13946001/presentation/
```

## Methods

| Method | Role | Idea |
|---|---|---|
| Sync FedAvg | baseline | Wait for all clients in each round. |
| Naive Async | baseline | Apply each arriving update immediately. |
| Staleness Async | baseline | Reduce update weight by logical staleness. |
| FedBuff-lite | baseline | Aggregate a buffer of async updates. |
| CAA-FedBuff | proposed v1 | Add direction agreement, clipping, and adaptive alpha. |
| CAA-v2 | final proposed method | Add server trajectory memory and client fairness credit. |

CAA-v2 is the final method because it is the clearest distributed-systems
tradeoff. It is not the most aggressive peak-accuracy method, but it is more
stable and more explainable than naive async or CAA-v1.

## CAA-v2 Algorithm

For each buffered client update:

```text
delta_i = client_model_i - model_at_client_start_i
tau_i   = server_version - client_start_version_i
```

CAA-v2 computes a server-side aggregation weight:

```text
raw_weight_i =
    num_examples_i
  * staleness_decay(tau_i)
  * agreement_factor_i
  * fairness_credit_i
```

where:

- `staleness_decay(tau_i)` downweights old updates.
- `agreement_factor_i` checks whether the update direction agrees with the
  buffer and the recent accepted server trajectory.
- `fairness_credit_i` reduces long-term domination by frequently arriving fast
  clients.
- adaptive server alpha becomes larger when agreement is high and smaller when
  mean staleness is high.

The rule is **clockless**: it uses logical version, delta direction, and client
contribution count, not physical synchronized time.

## What Is Existing vs What Is Ours

| Component | Source | Role |
|---|---|---|
| FedAvg | existing | Sync baseline. |
| FedAsync / staleness-aware AFL | existing | Async and staleness baselines. |
| FedBuff | existing | Buffered async baseline. |
| MedMNIST | existing benchmark | Biomedical image datasets. |
| ResNet18 / MobileNetV3 / small CNN | existing models | Classification backbones. |
| Clockless event-driven simulator | ours | Simulates async arrivals with logical versions. |
| Fair update-budget protocol | ours | Uses `async events = sync rounds * clients`. |
| CAA-FedBuff / CAA-v2 | ours | Agreement/fairness-aware buffered aggregation rules. |
| Distributed-systems analysis pipeline | ours | Staleness, simulated time, client Gini, stability drop. |

## Experimental Setup

Official headline matrix:

```text
9 MedMNIST datasets
6 methods
3 seeds per dataset/method
ResNet18
IID partition
fair update budget
```

Datasets:

```text
pathmnist, pneumoniamnist, bloodmnist, organamnist, organcmnist,
dermamnist, octmnist, breastmnist, tissuemnist
```

Fairness controls:

| Control | Setting |
|---|---|
| Clients | 10 |
| Local epochs | 1 |
| Batch size | 128 |
| LR scheduler | cosine, lr=0.01, min_lr=0.0001 |
| Async delay | same heterogeneous setting across async baselines |
| Budget | `async events = sync rounds * clients` |
| Seeds | 42, 43, 44 |

## Headline Results

Overall mean across 9 datasets:

| Method | Best Acc Mean | Final Acc Mean | Stability Drop |
|---|---:|---:|---:|
| Sync FedAvg | 0.7142 | 0.7121 | 0.0020 |
| Naive Async | 0.7132 | 0.7096 | 0.0036 |
| Staleness Async | 0.6770 | 0.6752 | 0.0017 |
| FedBuff-lite | 0.7090 | 0.7062 | 0.0028 |
| CAA-FedBuff | 0.7206 | 0.7158 | 0.0048 |
| CAA-v2 | 0.7169 | 0.7140 | 0.0029 |

Main findings:

- CAA-v2 beats Sync FedAvg in mean best accuracy on `6/9` datasets.
- CAA-v2 beats Sync FedAvg in mean final accuracy on `6/9` datasets.
- CAA-v2 beats the strongest classic baseline among Sync/Naive/Staleness/FedBuff
  on `5/9` datasets.
- Staleness-only aggregation is stable but too conservative.
- CAA-v1 has the strongest peak accuracy, but larger stability drop.
- CAA-v2 is the best final story because it balances performance, stability,
  and distributed-systems interpretability.

Recommended claim:

> Under a fair update budget, CAA-v2 makes clockless asynchronous FL approach
> Sync FedAvg across diverse MedMNIST datasets, while reducing the instability
> of naive async and avoiding the over-conservatism of staleness-only
> aggregation.

## Evaluation Metrics

We report more than accuracy:

| Metric | Meaning |
|---|---|
| Async-Sync Best Gap | Peak accuracy cost of removing the sync barrier. |
| Async-Sync Final Gap | Whether async converges near sync at the end. |
| Stability Drop | `best_acc - final_acc`; oscillation after the best point. |
| p95 staleness | Tail logical delay under async arrivals. |
| client contribution Gini | Whether fast clients dominate accepted updates. |
| time-to-accuracy | Simulated time needed to approach best performance. |

## Deliverables

Implemented:

- CSV logging and summary JSON for every run.
- Best checkpoint saving.
- General staleness decay functions.
- Event-driven async simulation with heterogeneous delays.
- FedBuff-lite and CAA-family methods.
- Multi-dataset MedMNIST support.
- Dirichlet non-IID partitions.
- ResNet18, small CNN, and MobileNetV3-small support.
- Plotting and report summary scripts.
- Chinese and English presentation materials.

## How To Run

Install:

```bash
cd r13946001/pathMNIST
python -m pip install -e .
```

Smoke test:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py \
  --synthetic --method caa_fedbuff_v2 \
  --events 4 --clients 2 --buffer-size 2 \
  --model small_cnn --device cpu
```

Official-style PathMNIST run:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py \
  --dataset pathmnist --method caa_fedbuff_v2 \
  --events 1000 --clients 10 --buffer-size 5 \
  --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 \
  --local-epochs 1 --augment --delay-mode heterogeneous \
  --device cuda --save-best
```

Regenerate report:

```bash
PYTHONPATH=src python -m fed_pathmnist.plot_results --csv results/*.csv --outdir figures
python scripts/plot_report_summary.py --result-dir results --outdir figures/report
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report
python scripts/plot_distributed_systems_summary.py --result-dir results --outdir figures/report
python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md
```

## Novelty Boundary

CAA-v2 has course-project and workshop/demo value, especially as a
distributed-systems implementation and evaluation. It should not be overstated
as a new state-of-the-art FL algorithm. Its value is in:

- the clockless framing,
- the explicit aggregation rule,
- the fair update-budget comparison,
- the multi-dataset/multi-seed study,
- the system metrics beyond accuracy.

## Privacy Boundary

The project keeps raw images local in the federated-learning simulation. It does
not yet implement secure aggregation, differential privacy, or real hospital
deployment. Those are natural future extensions.
