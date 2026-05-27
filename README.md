# Clockless Federated Adaptation

**Agreement-Aware Asynchronous Federated Learning for Medical Imaging**

This repository is the final project for Group 4, originally titled
**Asynchronous Distributed ML Adaptation**.  The current project studies a
distributed-systems question:

> In a federated learning system without a global clock, can asynchronous
> training approach Sync FedAvg while controlling stale updates, stragglers, and
> fast-client domination?

The project is implemented as a reproducible research/demo codebase, not as a
production federated learning framework.

## Team

| Name | Student ID | Department |
|---|---|---|
| 陳冠宇 | R13946001 | Data Science |
| 張光澄 | R14922172 | Computer Science |
| 張育嘉 | R14922140 | Computer Science |

## Motivation

Medical AI is a natural distributed learning scenario. Hospitals may want to
train a shared model without centralizing raw images, but each hospital has
different hardware, network delay, workload, and patient distribution.

Synchronous FL is stable, but every round waits for slow clients. Asynchronous
FL improves throughput, but a late update may have been trained from an older
global model. This creates the core problem of **staleness**.

We model time using logical model versions instead of synchronized physical
clocks:

```text
staleness = current_server_version - client_start_version
```

This makes the project a distributed-systems study of event ordering,
staleness, stragglers, and fairness, with medical image classification as the
experimental workload.

## Methods

Implemented methods:

| Method | Role | Main idea |
|---|---|---|
| Sync FedAvg | baseline | Barrier synchronization; stable but waits for slow clients. |
| Naive Async | baseline | Applies each arriving update immediately with constant alpha. |
| Staleness Async | baseline | Downweights updates by logical staleness. |
| FedBuff-lite | baseline | Buffers async updates before aggregation. |
| CAA-FedBuff | proposed v1 | Adds direction agreement, clipping, and adaptive alpha. |
| CAA-v2 | final proposed method | Adds server trajectory agreement and client fairness credit. |

CAA-v2 is a deterministic course-project design extension. It is not claimed as
a publication-level new FL algorithm. Its contribution is to combine
server-observable, clockless signals in a simple aggregation rule:

```text
raw_weight_i =
    num_examples_i
  * staleness_decay(tau_i)
  * agreement_factor_i
  * fairness_credit_i
```

where `tau_i` is logical staleness, `agreement_factor_i` measures whether an
update direction agrees with the buffered/server trajectory direction, and
`fairness_credit_i` reduces long-term fast-client domination.

## Experimental Scope

Official headline results use a fair multi-seed matrix:

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
pathmnist, pneumoniamnist, bloodmnist, organamnist, organcmnist,
dermamnist, octmnist, breastmnist, tissuemnist
```

The project also includes non-IID Dirichlet experiments, straggler/delay stress
tests, CAA-v2 ablations, multi-backbone support, and classification diagnostics.

## Headline Results

Overall method performance across the 9-dataset fair matrix:

| Method | Best Acc Mean | Final Acc Mean | Stability Drop Mean |
|---|---:|---:|---:|
| Sync FedAvg | 0.7142 | 0.7121 | 0.0020 |
| Naive Async | 0.7132 | 0.7096 | 0.0036 |
| Staleness Async | 0.6770 | 0.6752 | 0.0017 |
| FedBuff-lite | 0.7090 | 0.7062 | 0.0028 |
| CAA-FedBuff | 0.7206 | 0.7158 | 0.0048 |
| CAA-v2 | 0.7169 | 0.7140 | 0.0029 |

Interpretation:

- CAA-v2 beats Sync FedAvg in mean best accuracy on `6/9` datasets and mean
  final accuracy on `6/9` datasets.
- CAA-v2 beats the strongest classic baseline among Sync/Naive/Staleness/FedBuff
  on `5/9` datasets.
- CAA-FedBuff v1 has the highest mean peak accuracy, but it is less stable.
- Staleness-only aggregation is stable but too conservative.
- CAA-v2 is the cleaner final method because it trades a little peak accuracy
  for better final accuracy and lower oscillation than Naive Async / CAA-v1.

Conservative final claim:

> Under a fair update budget, CAA-v2 makes clockless asynchronous FL approach
> Sync FedAvg across diverse MedMNIST datasets, while reducing the instability
> of naive async and avoiding the over-conservatism of staleness-only
> aggregation.

## Repository Layout

```text
pathMNIST/                    Original team baseline code
r13946001/                    Safe experiment copy and final implementation
  pathMNIST/                  Main runnable codebase
    src/fed_pathmnist/        Dataset, model, simulator, CLI, plotting
    scripts/                  Experiment and reporting utilities
  REPORT_NOTES.md             Detailed experiment notes and tables
  FINAL_RESULTS_SUMMARY.md    Clean final result summary
  NOVELTY_ASSESSMENT.md       Research positioning and novelty boundary
  presentation/               Proposal/report slides and speaker notes
PP-DMA：Privacy-Preserving Distributed Model Adaptation.md
                              Project brief / proposal summary
```

Generated outputs such as `results/`, `figures/`, `logs/`, `checkpoints/`, and
`data/` are ignored by git.

## Quick Start

Use Python 3.10+ and install the project dependencies:

```bash
cd r13946001/pathMNIST
python -m pip install -e .
```

Run a small CPU smoke test:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py \
  --synthetic --method caa_fedbuff_v2 \
  --events 4 --clients 2 --buffer-size 2 \
  --model small_cnn --device cpu
```

Run one official-style PathMNIST CAA-v2 experiment:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py \
  --dataset pathmnist --method caa_fedbuff_v2 \
  --events 1000 --clients 10 --buffer-size 5 \
  --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 \
  --local-epochs 1 --augment --delay-mode heterogeneous \
  --device cuda --save-best
```

Regenerate report artifacts from existing results:

```bash
PYTHONPATH=src python -m fed_pathmnist.plot_results --csv results/*.csv --outdir figures
python scripts/plot_report_summary.py --result-dir results --outdir figures/report
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report
python scripts/plot_distributed_systems_summary.py --result-dir results --outdir figures/report
python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md
```

## Presentation Materials

The polished presentation materials are under:

```text
r13946001/presentation/
```

Important files:

- `clockless_federated_adaptation_proposal_zh.pptx`
- `clockless_federated_adaptation_proposal_en.pptx`
- `speaker_notes_zh.md`
- `speaker_notes_en.md`

## Privacy Note

This project is privacy-preserving in the federated-learning sense that raw
medical images stay local to clients in the simulation. It does **not** yet
implement cryptographic secure aggregation, differential privacy, or a real
multi-hospital deployment. Those are listed as future work.
