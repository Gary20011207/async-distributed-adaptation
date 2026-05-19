# R13946001 Experiment Notes

## Story for Presentation

Different hospitals and edge devices do not train at the same speed. Some nodes return fresh updates, while stragglers return stale updates computed from older global models.

The distributed-systems question is not only accuracy. It is how a system without a global clock should reason about delayed information. This project uses logical model versions to define staleness and compares aggregation policies under asynchronous arrivals.

## Headline Findings

- Fair async comparison uses the same client-update budget: `async events = sync rounds * clients`.
- Best CAA-family run reached `0.9145`, compared with strongest stateless async `0.9162` and Sync FedAvg `0.9110`.
- In the current completed runs, CAA-family methods do not beat every baseline; use the gap/stability analysis below.
- Logical staleness alone can be conservative: it reduces stale-update impact, but may also shrink useful updates too much.
- Completed datasets in this report: `bloodmnist, breastmnist, dermamnist, octmnist, organamnist, organcmnist, pathmnist, pneumoniamnist, tissuemnist`.
- The main report metrics are `Async-Sync Best Gap`, `Async-Sync Final Gap`, and `Stability Drop = best_acc - final_acc`.

## Method Detail

- `sync_fedavg`: barrier baseline; the server waits for all clients each round.
- `naive_async`: stateless async baseline; every arriving update is applied with constant alpha.
- `staleness_async`: logical-staleness baseline; alpha is reduced by `server_version - client_start_version`.
- `fedbuff_async`: buffered async baseline; `B` stale/data-size weighted deltas are aggregated together.
- `agreement_fedbuff_async`: CAA-FedBuff adds direction agreement, median-norm clipping, and adaptive server alpha.
- `caa_fedbuff_v2`: CAA-v2 additionally compares updates with recent accepted server delta direction and adds client fairness credit.

CAA-FedBuff and CAA-v2 are AI-assisted implemented course-project algorithms, but their aggregation rules are explicit and deterministic. It combines known ideas from buffered async FL, staleness-aware aggregation, cosine agreement, clipping, and adaptive server step size. We claim it as our own implemented design extension, not as a publication-level novel FL algorithm.

Simplified CAA-v2 rule: buffered update weight = data size x staleness decay x direction agreement x fairness credit; server alpha increases with agreement and decreases with mean staleness. The method remains clockless because it uses logical versions, deltas, and client contribution counts rather than a physical global clock.

## Existing vs Ours

| Component | Source | Role in this project |
|---|---|---|
| Sync FedAvg | existing baseline | Barrier aggregation; server waits for every client in each round. |
| Naive Async | existing baseline | Applies each arriving client update immediately with constant alpha. |
| Staleness-aware decay | existing baseline | Uses logical version gap to reduce stale update impact. |
| FedBuff-style buffering | existing baseline | Aggregates a buffer of asynchronous updates instead of one update at a time. |
| MedMNIST benchmark | existing benchmark | Medical image classification datasets used to evaluate the distributed-learning setting. |
| ResNet18 / MobileNetV3 | existing backbone | Standard image classifiers used as model backbones. |
| Clockless simulator and logging | our implementation | Event-driven async simulator with logical versions, simulated time, CSV summaries, and plots. |
| CAA agreement weighting | our design | Weights buffered deltas by direction agreement without using a physical global clock. |
| CAA-v2 server trajectory EMA | our design | Compares updates with recent accepted server direction to reject conflicting movement. |
| CAA-v2 client fairness credit | our design | Reduces domination by frequently arriving fast clients using only client ids and contribution counts. |
| Fair-budget analysis pipeline | our implementation | Compares sync rounds and async events under the same client-update budget with multi-seed reports. |

## Fairness Protocol

| Control | Value |
|---|---|
| Clients | 10 |
| Local epochs | 1 |
| Batch size | 128 |
| LR schedule | cosine, lr=0.01, min_lr=0.0001 |
| Augmentation | enabled for official runs |
| Partition | IID unless explicitly marked Dirichlet |
| Async delay | heterogeneous with shared straggler settings across async methods |
| Fair budget | `async events = sync rounds * clients` |
| Seed | controls split, partition, delay sampling, and initialization |

Headline comparisons should not mix different dataset sample limits, backbones, seeds, local epochs, delay distributions, or update budgets.

## Multi-Seed Variance

| Dataset | Model | Method | Seeds | Best Acc Mean | Best Acc Std | Final Acc Mean | Final Acc Std | Stability Drop Mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bloodmnist | resnet18 | CAA | 3 | 0.8810 | 0.0064 | 0.8792 | 0.0075 | 0.0019 |
| bloodmnist | resnet18 | CAA-v2 | 3 | 0.8765 | 0.0076 | 0.8752 | 0.0069 | 0.0014 |
| bloodmnist | resnet18 | Naive | 3 | 0.8719 | 0.0064 | 0.8704 | 0.0067 | 0.0015 |
| bloodmnist | resnet18 | Staleness | 3 | 0.8810 | 0.0098 | 0.8793 | 0.0102 | 0.0018 |
| bloodmnist | resnet18 | Sync | 3 | 0.8754 | 0.0084 | 0.8753 | 0.0083 | 0.0001 |
| breastmnist | resnet18 | CAA | 1 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | CAA-v2 | 1 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | Naive | 1 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | Staleness | 1 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | Sync | 1 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| dermamnist | resnet18 | CAA | 1 | 0.6898 | 0.0000 | 0.6888 | 0.0000 | 0.0010 |
| dermamnist | resnet18 | Naive | 1 | 0.6918 | 0.0000 | 0.6908 | 0.0000 | 0.0010 |
| dermamnist | resnet18 | Staleness | 1 | 0.6903 | 0.0000 | 0.6898 | 0.0000 | 0.0005 |
| dermamnist | resnet18 | Sync | 1 | 0.6928 | 0.0000 | 0.6903 | 0.0000 | 0.0025 |
| octmnist | resnet18 | CAA | 1 | 0.5430 | 0.0000 | 0.5250 | 0.0000 | 0.0180 |
| octmnist | resnet18 | Naive | 1 | 0.5320 | 0.0000 | 0.5190 | 0.0000 | 0.0130 |
| octmnist | resnet18 | Staleness | 1 | 0.5400 | 0.0000 | 0.5320 | 0.0000 | 0.0080 |
| octmnist | resnet18 | Sync | 1 | 0.5320 | 0.0000 | 0.5290 | 0.0000 | 0.0030 |
| organamnist | resnet18 | CAA | 3 | 0.6348 | 0.0079 | 0.6314 | 0.0065 | 0.0034 |
| organamnist | resnet18 | CAA-v2 | 3 | 0.6248 | 0.0073 | 0.6233 | 0.0059 | 0.0015 |
| organamnist | resnet18 | Naive | 3 | 0.6209 | 0.0092 | 0.6139 | 0.0099 | 0.0070 |
| organamnist | resnet18 | Staleness | 3 | 0.6266 | 0.0098 | 0.6213 | 0.0089 | 0.0053 |
| organamnist | resnet18 | Sync | 3 | 0.6185 | 0.0079 | 0.6157 | 0.0059 | 0.0028 |
| organcmnist | resnet18 | CAA-v2 | 1 | 0.6542 | 0.0000 | 0.6410 | 0.0000 | 0.0132 |
| organcmnist | resnet18 | Naive | 1 | 0.6365 | 0.0000 | 0.6355 | 0.0000 | 0.0010 |
| organcmnist | resnet18 | Staleness | 1 | 0.6502 | 0.0000 | 0.6500 | 0.0000 | 0.0002 |
| organcmnist | resnet18 | Sync | 1 | 0.6428 | 0.0000 | 0.6428 | 0.0000 | 0.0000 |
| pathmnist | resnet18 | CAA | 3 | 0.8968 | 0.0112 | 0.8928 | 0.0114 | 0.0040 |
| pathmnist | resnet18 | CAA-v2 | 3 | 0.9049 | 0.0098 | 0.9023 | 0.0092 | 0.0026 |
| pathmnist | resnet18 | FedBuff | 1 | 0.8854 | 0.0000 | 0.8838 | 0.0000 | 0.0015 |
| pathmnist | resnet18 | Naive | 3 | 0.8985 | 0.0161 | 0.8921 | 0.0132 | 0.0064 |
| pathmnist | resnet18 | Staleness | 3 | 0.8989 | 0.0078 | 0.8909 | 0.0090 | 0.0079 |
| pathmnist | resnet18 | Sync | 3 | 0.8998 | 0.0103 | 0.8935 | 0.0095 | 0.0062 |
| pneumoniamnist | resnet18 | CAA | 3 | 0.8707 | 0.0076 | 0.8552 | 0.0052 | 0.0155 |
| pneumoniamnist | resnet18 | CAA-v2 | 3 | 0.8606 | 0.0028 | 0.8542 | 0.0028 | 0.0064 |
| pneumoniamnist | resnet18 | Naive | 3 | 0.8515 | 0.0009 | 0.8515 | 0.0009 | 0.0000 |
| pneumoniamnist | resnet18 | Staleness | 3 | 0.8574 | 0.0048 | 0.8552 | 0.0082 | 0.0021 |
| pneumoniamnist | resnet18 | Sync | 3 | 0.8531 | 0.0024 | 0.8515 | 0.0037 | 0.0016 |
| tissuemnist | resnet18 | CAA | 1 | 0.5622 | 0.0000 | 0.5618 | 0.0000 | 0.0004 |
| tissuemnist | resnet18 | CAA-v2 | 1 | 0.5564 | 0.0000 | 0.5564 | 0.0000 | 0.0000 |
| tissuemnist | resnet18 | Naive | 1 | 0.5586 | 0.0000 | 0.5558 | 0.0000 | 0.0028 |
| tissuemnist | resnet18 | Staleness | 1 | 0.5584 | 0.0000 | 0.5580 | 0.0000 | 0.0004 |
| tissuemnist | resnet18 | Sync | 1 | 0.5564 | 0.0000 | 0.5564 | 0.0000 | 0.0000 |

## Best-by-Dataset View

For each dataset and method, this table reports the best completed full run. Negative gaps mean async matched or exceeded Sync FedAvg in that run.

| Dataset | Sync Best | Stateless Best | Staleness Best | FedBuff Best | CAA Best | CAA-v2 Best | Best CAA-Sync Gap | Best CAA Stability Drop | Best Async Method |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bloodmnist | 0.8825 | 0.8787 | 0.8904 |  | 0.8857 | 0.8822 | -0.0032 | 0.0023 | staleness_async |
| breastmnist | 0.7308 | 0.7308 | 0.7308 |  | 0.7308 | 0.7308 | 0.0000 | 0.0000 | naive_async |
| dermamnist | 0.6928 | 0.6918 | 0.6903 |  | 0.6898 |  | 0.0030 | 0.0010 | naive_async |
| octmnist | 0.5320 | 0.5320 | 0.5400 |  | 0.5430 |  | -0.0110 | 0.0180 | agreement_fedbuff_async |
| organamnist | 0.6265 | 0.6308 | 0.6362 |  | 0.6402 | 0.6322 | -0.0138 | 0.0102 | agreement_fedbuff_async |
| organcmnist | 0.6428 | 0.6365 | 0.6502 |  |  | 0.6542 | -0.0115 | 0.0132 | caa_fedbuff_v2 |
| pathmnist | 0.9110 | 0.9162 | 0.9063 | 0.8976 | 0.9118 | 0.9145 | -0.0035 | 0.0043 | naive_async |
| pneumoniamnist | 0.8558 | 0.8526 | 0.8622 |  | 0.8766 | 0.8622 | -0.0208 | 0.0192 | agreement_fedbuff_async |
| tissuemnist | 0.5564 | 0.5586 | 0.5584 |  | 0.5622 | 0.5564 | -0.0058 | 0.0004 | agreement_fedbuff_async |

## Detailed Result Summary

This table keeps all completed full runs, including tuning runs. Use the best-by-dataset view above for slides.

| Dataset | Model | Run | Budget | Best Acc | Best Step | Final Acc | Final Loss | Progress | Sim Time | Avg Staleness | Avg Alpha | Avg Agreement | Buffer Alpha | Dropped | Client Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bloodmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8837 | 300 | 0.8837 | 0.3359 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7183 | 0.6263 | 0 | 5-38 |
| bloodmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8857 | 280 | 0.8834 | 0.3274 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7193 | 0.6256 | 1 | 5-41 |
| bloodmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8737 | 280 | 0.8705 | 0.3460 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7390 | 0.6274 | 0 | 7-38 |
| breastmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.7308 | 1 | 0.7308 | 0.5562 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.8187 | 0.6387 | 0 | 5-38 |
| dermamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6898 | 240 | 0.6888 | 0.8473 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6972 | 0.6260 | 2 | 5-38 |
| octmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5430 | 280 | 0.5250 | 1.1417 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7235 | 0.6277 | 1 | 5-38 |
| organamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6258 | 300 | 0.6258 | 0.9043 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7443 | 0.6304 | 1 | 5-38 |
| organamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6385 | 300 | 0.6385 | 0.8725 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7511 | 0.6298 | 1 | 5-41 |
| organamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6402 | 280 | 0.6300 | 0.8883 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7341 | 0.6267 | 0 | 7-38 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 500 | 0.8825 | 340 | 0.8801 | 0.5754 | 500 | 181.0887 | 1.7720 | 0.5465 | 0.5615 | 0.5415 | 0 | 10-63 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 500 | 0.8858 | 340 | 0.8815 | 0.5515 | 500 | 181.0887 | 1.7720 | 0.5962 | 0.5723 | 0.5903 | 0 | 10-63 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=3 | 500 | 0.8708 | 280 | 0.1864 | nan | 500 | 181.0887 | 2.9580 | 0.5428 | 0.7823 | 0.5296 | 12 | 10-63 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / polynomial / B=5 | 500 | 0.8848 | 340 | 0.8797 | 0.5562 | 500 | 181.0887 | 1.7720 | 0.4425 | 0.5492 | 0.5395 | 0 | 10-63 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 1000 | 0.9040 | 860 | 0.9000 | 0.3693 | 1000 | 357.5957 | 1.7850 | 0.5960 | 0.5425 | 0.5856 | 0 | 20-126 |
| pathmnist | resnet18 | agreement_fedbuff_async / dirichlet / hinge / B=5 | 500 | 0.8709 | 480 | 0.8454 | 0.4994 | 500 | 181.0887 | 1.7720 | 0.5962 | 0.5279 | 0.5880 | 6 | 10-63 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=3 | 1000 | 0.4735 | 140 | 0.1864 | nan | 1000 | 357.5957 | 2.9780 | 0.6413 | 0.8373 | 0.6326 | 18 | 20-126 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 1000 | 0.9097 | 600 | 0.9011 | 0.3710 | 1000 | 357.5957 | 1.7850 | 0.6456 | 0.5622 | 0.6474 | 1 | 20-126 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 1000 | 0.9118 | 700 | 0.9019 | 0.4226 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5462 | 0.6002 | 0 | 20-126 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=7 | 1000 | 0.8986 | 620 | 0.8928 | 0.4737 | 1000 | 357.5957 | 1.2730 | 0.6478 | 0.4748 | 0.6451 | 0 | 20-126 |
| pathmnist | resnet18 | agreement_fedbuff_async / dirichlet / hinge / B=5 | 500 | 0.8737 | 360 | 0.8540 | 0.4673 | 500 | 181.0887 | 1.7720 | 0.6161 | 0.4739 | 0.5912 | 0 | 10-63 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 1000 | 0.8964 | 660 | 0.8926 | 0.4220 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5400 | 0.5995 | 0 | 22-129 |
| pathmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 1000 | 0.9082 | 780 | 0.9042 | 0.3951 | 1000 | 353.5214 | 1.7800 | 0.6161 | 0.5488 | 0.6009 | 0 | 23-127 |
| pneumoniamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8734 | 160 | 0.8494 | 0.3826 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6719 | 0.6198 | 0 | 5-38 |
| pneumoniamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8622 | 180 | 0.8590 | 0.3578 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6580 | 0.6169 | 0 | 5-41 |
| pneumoniamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8766 | 160 | 0.8574 | 0.3893 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6674 | 0.6177 | 0 | 7-38 |
| tissuemnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5622 | 280 | 0.5618 | 1.1769 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6786 | 0.6209 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8796 | 280 | 0.8766 | 0.3480 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7184 | 0.6095 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8796 | 280 | 0.8784 | 0.3449 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6846 | 0.6048 | 0 | 5-41 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8679 | 280 | 0.8673 | 0.3559 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7043 | 0.6068 | 0 | 7-38 |
| bloodmnist | small_cnn | caa_fedbuff_v2 / iid | 300 | 0.7778 | 280 | 0.7726 | 0.6061 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7926 | 0.6204 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8822 | 280 | 0.8799 | 0.3370 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7314 | 0.6279 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8790 | 280 | 0.8769 | 0.3454 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7026 | 0.6077 | 0 | 5-38 |
| breastmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5561 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.8063 | 0.6346 | 0 | 5-38 |
| breastmnist | mobilenet_v3_small | caa_fedbuff_v2 / iid | 300 | 0.7308 | 20 | 0.7308 | 0.6735 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.9947 | 0.6614 | 0 | 5-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6178 | 300 | 0.6178 | 0.9276 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7240 | 0.6104 | 0 | 5-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6245 | 280 | 0.6228 | 0.8918 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7053 | 0.6072 | 0 | 5-41 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6322 | 280 | 0.6295 | 0.8987 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7165 | 0.6093 | 1 | 7-38 |
| organamnist | small_cnn | caa_fedbuff_v2 / iid | 300 | 0.5757 | 260 | 0.5737 | 1.0631 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7382 | 0.6139 | 0 | 5-38 |
| organcmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6542 | 280 | 0.6410 | 0.8989 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6933 | 0.6069 | 0 | 5-38 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9049 | 700 | 0.9007 | 0.3858 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5500 | 0.5854 | 0 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8950 | 540 | 0.8922 | 0.4519 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5500 | 0.5856 | 0 | 22-129 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9145 | 820 | 0.9102 | 0.3336 | 1000 | 353.5214 | 1.7800 | 0.6161 | 0.5456 | 0.5854 | 1 | 23-127 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9053 | 860 | 0.9046 | 0.3517 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5477 | 0.6004 | 0 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9008 | 600 | 0.8961 | 0.3756 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5346 | 0.5837 | 0 | 20-126 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8574 | 140 | 0.8526 | 0.3858 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6127 | 0.6021 | 0 | 5-38 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8622 | 200 | 0.8574 | 0.3569 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6054 | 0.6001 | 0 | 5-41 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8622 | 160 | 0.8526 | 0.3879 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6026 | 0.5989 | 0 | 7-38 |
| pneumoniamnist | small_cnn | caa_fedbuff_v2 / iid | 300 | 0.8429 | 80 | 0.8093 | 0.4258 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6951 | 0.6156 | 0 | 5-38 |
| tissuemnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5564 | 300 | 0.5564 | 1.1855 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6650 | 0.6024 | 0 | 5-38 |
| pathmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 500 | 0.8854 | 340 | 0.8838 | 0.5395 | 500 | 181.0887 | 1.7720 | 0.2264 |  |  |  | 10-63 |
| pathmnist | resnet18 | fedbuff_async / dirichlet / inverse / B=5 | 500 | 0.8735 | 240 | 0.8550 | 0.4731 | 500 | 181.0887 | 1.7720 | 0.2264 |  |  |  | 10-63 |
| pathmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 1000 | 0.8976 | 560 | 0.8923 | 0.4218 | 1000 | 357.5957 | 1.7850 | 0.2224 |  |  |  | 20-126 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8708 | 260 | 0.8670 | 0.3671 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8787 | 260 | 0.8781 | 0.3486 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8661 | 300 | 0.8661 | 0.3788 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| bloodmnist | small_cnn | naive_async / iid | 300 | 0.7808 | 260 | 0.7676 | 0.6225 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| breastmnist | resnet18 | naive_async / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5767 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| breastmnist | mobilenet_v3_small | naive_async / iid | 300 | 0.7308 | 1 | 0.7308 | 0.6792 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| dermamnist | resnet18 | naive_async / iid | 300 | 0.6918 | 220 | 0.6908 | 0.8468 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| octmnist | resnet18 | naive_async / iid | 300 | 0.5320 | 240 | 0.5190 | 1.1561 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6125 | 260 | 0.6040 | 0.9419 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6308 | 240 | 0.6238 | 0.9125 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6195 | 220 | 0.6140 | 0.9325 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| organamnist | small_cnn | naive_async / iid | 300 | 0.5753 | 300 | 0.5753 | 1.0713 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organcmnist | resnet18 | naive_async / iid | 300 | 0.6365 | 260 | 0.6355 | 0.9082 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| pathmnist | resnet18 | naive_async / iid | 500 | 0.8847 | 320 | 0.8806 | 0.5442 | 500 | 181.0887 | 8.8920 | 0.5000 |  |  |  | 10-63 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.9049 | 700 | 0.8961 | 0.4505 | 1000 | 357.5957 | 8.9430 | 0.5000 |  |  |  | 20-126 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.8946 | 600 | 0.8891 | 0.4805 | 1000 | 355.3563 | 8.9030 | 0.5000 |  |  |  | 22-129 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.9162 | 480 | 0.9065 | 0.3360 | 1000 | 353.5214 | 8.9210 | 0.5000 |  |  |  | 23-127 |
| pneumoniamnist | resnet18 | naive_async / iid | 300 | 0.8526 | 260 | 0.8526 | 0.3845 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | naive_async / iid | 300 | 0.8510 | 300 | 0.8510 | 0.3687 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| pneumoniamnist | resnet18 | naive_async / iid | 300 | 0.8510 | 300 | 0.8510 | 0.3791 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| pneumoniamnist | small_cnn | naive_async / iid | 300 | 0.7949 | 300 | 0.7949 | 0.4399 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| tissuemnist | resnet18 | naive_async / iid | 300 | 0.5586 | 260 | 0.5558 | 1.1963 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8819 | 260 | 0.8813 | 0.3306 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8904 | 260 | 0.8883 | 0.3172 | 300 | 109.0030 | 8.7367 | 0.3959 |  |  |  | 5-41 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8708 | 220 | 0.8682 | 0.3540 | 300 | 104.8960 | 8.7800 | 0.3965 |  |  |  | 7-38 |
| bloodmnist | small_cnn | staleness_async / iid / hinge | 300 | 0.7948 | 240 | 0.7919 | 0.5799 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| breastmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.7308 | 1 | 0.7308 | 0.5720 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| breastmnist | mobilenet_v3_small | staleness_async / iid / hinge | 300 | 0.7308 | 1 | 0.7308 | 0.6771 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| dermamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6903 | 160 | 0.6898 | 0.8378 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| octmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.5400 | 240 | 0.5320 | 1.1293 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6168 | 280 | 0.6125 | 0.9134 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6362 | 240 | 0.6302 | 0.8850 | 300 | 109.0030 | 8.7367 | 0.3959 |  |  |  | 5-41 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6268 | 260 | 0.6212 | 0.9008 | 300 | 104.8960 | 8.7800 | 0.3965 |  |  |  | 7-38 |
| organamnist | small_cnn | staleness_async / iid / hinge | 300 | 0.5825 | 300 | 0.5825 | 1.0362 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organcmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6502 | 280 | 0.6500 | 0.8787 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| pathmnist | resnet18 | staleness_async / iid / inverse | 500 | 0.8673 | 380 | 0.8627 | 0.5351 | 500 | 181.0887 | 8.8920 | 0.0725 |  |  |  | 10-63 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 500 | 0.8908 | 320 | 0.8812 | 0.5780 | 500 | 181.0887 | 8.8920 | 0.3941 |  |  |  | 10-63 |
| pathmnist | resnet18 | staleness_async / dirichlet / inverse | 500 | 0.8519 | 360 | 0.8242 | 0.5235 | 500 | 181.0887 | 8.8920 | 0.0725 |  |  |  | 10-63 |
| pathmnist | resnet18 | staleness_async / iid / inverse | 1000 | 0.8882 | 680 | 0.8788 | 0.4687 | 1000 | 357.5957 | 8.9430 | 0.0710 |  |  |  | 20-126 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.8996 | 660 | 0.8926 | 0.3988 | 1000 | 355.3563 | 8.9030 | 0.3942 |  |  |  | 22-129 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.9063 | 660 | 0.8990 | 0.3824 | 1000 | 353.5214 | 8.9210 | 0.3939 |  |  |  | 23-127 |
| pneumoniamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8526 | 200 | 0.8462 | 0.3835 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8622 | 300 | 0.8622 | 0.3519 | 300 | 109.0030 | 8.7367 | 0.3959 |  |  |  | 5-41 |
| pneumoniamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8574 | 220 | 0.8574 | 0.3814 | 300 | 104.8960 | 8.7800 | 0.3965 |  |  |  | 7-38 |
| pneumoniamnist | small_cnn | staleness_async / iid / hinge | 300 | 0.8141 | 220 | 0.8093 | 0.4231 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| tissuemnist | resnet18 | staleness_async / iid / hinge | 300 | 0.5584 | 260 | 0.5580 | 1.1850 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| bloodmnist | resnet18 | sync_fedavg / iid | 300 | 0.8775 | 29 | 0.8775 | 0.3535 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | resnet18 | sync_fedavg / iid | 300 | 0.8825 | 29 | 0.8822 | 0.3368 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | resnet18 | sync_fedavg / iid | 300 | 0.8661 | 30 | 0.8661 | 0.3681 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | small_cnn | sync_fedavg / iid | 300 | 0.7790 | 29 | 0.7790 | 0.5964 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| breastmnist | resnet18 | sync_fedavg / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5776 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| breastmnist | mobilenet_v3_small | sync_fedavg / iid | 300 | 0.7308 | 1 | 0.7308 | 0.6783 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| dermamnist | resnet18 | sync_fedavg / iid | 300 | 0.6928 | 24 | 0.6903 | 0.8430 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| octmnist | resnet18 | sync_fedavg / iid | 300 | 0.5320 | 26 | 0.5290 | 1.1224 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / iid | 300 | 0.6108 | 27 | 0.6095 | 0.9239 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / iid | 300 | 0.6265 | 23 | 0.6212 | 0.9030 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / iid | 300 | 0.6182 | 22 | 0.6162 | 0.9135 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | small_cnn | sync_fedavg / iid | 300 | 0.5747 | 27 | 0.5747 | 1.0575 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organcmnist | resnet18 | sync_fedavg / iid | 300 | 0.6428 | 26 | 0.6428 | 0.9065 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 500 | 0.8907 | 32 | 0.8857 | 0.4305 | 50 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / dirichlet | 500 | 0.8762 | 37 | 0.8708 | 0.4159 | 50 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 1000 | 0.9032 | 65 | 0.8953 | 0.3849 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 1000 | 0.8976 | 78 | 0.8909 | 0.4495 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 1000 | 0.9110 | 75 | 0.9040 | 0.3566 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / iid | 300 | 0.8526 | 17 | 0.8494 | 0.3779 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / iid | 300 | 0.8558 | 30 | 0.8558 | 0.3668 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / iid | 300 | 0.8510 | 18 | 0.8494 | 0.3861 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | small_cnn | sync_fedavg / iid | 300 | 0.7997 | 29 | 0.7997 | 0.4292 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| tissuemnist | resnet18 | sync_fedavg / iid | 300 | 0.5564 | 30 | 0.5564 | 1.1946 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |

## CAA-Family Check

- Best CAA-family run: `caa_fedbuff_v2 / iid` at `0.9145`.
- Strongest non-CAA baseline: `naive_async / iid` at `0.9162`.
- Conclusion: The CAA-family method did not beat the strongest completed non-CAA baseline; report the stability/behavioral trade-off honestly.

## Agreement Analysis

- Best CAA-family run: `caa_fedbuff_v2 / iid`.
- Average positive agreement was `0.5456`; higher values mean buffered client deltas pointed in a similar direction.
- Average adaptive buffer alpha was `0.5854`, showing how the server adjusted step size from agreement and staleness.
- Dropped stale/conflicting updates: `1`. A low number means CAA mainly reweighted updates rather than filtering many clients.
- Use these metrics to explain whether agreement made the async path smoother, more aggressive, or too conservative.

## Async-Sync Gap Analysis

Sync FedAvg is the accuracy/stability reference. Async methods should be judged by how close they can get while avoiding the synchronization barrier.

| Run | Sync Ref | Best Gap | Final Gap | Stability Drop |
|---|---:|---:|---:|---:|
| bloodmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0012 | -0.0015 | 0.0000 |
| bloodmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0032 | -0.0012 | 0.0023 |
| bloodmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0088 | 0.0117 | 0.0032 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0029 | 0.0056 | 0.0029 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0029 | 0.0038 | 0.0012 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0146 | 0.0149 | 0.0006 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.1046 | 0.1096 | 0.0053 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0003 | 0.0023 | 0.0023 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0035 | 0.0053 | 0.0020 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0117 | 0.0152 | 0.0038 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0038 | 0.0041 | 0.0006 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0164 | 0.0161 | 0.0000 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.1017 | 0.1146 | 0.0132 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0006 | 0.0009 | 0.0006 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0079 | -0.0061 | 0.0020 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0117 | 0.0140 | 0.0026 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0877 | 0.0903 | 0.0029 |
| breastmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| dermamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0030 | 0.0015 | 0.0010 |
| dermamnist / naive_async / iid | sync_fedavg / iid | 0.0010 | -0.0005 | 0.0010 |
| dermamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0025 | 0.0005 | 0.0005 |
| octmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0110 | 0.0040 | 0.0180 |
| octmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0100 | 0.0130 |
| octmnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0080 | -0.0030 | 0.0080 |
| organamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0007 | -0.0045 | 0.0000 |
| organamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0120 | -0.0172 | 0.0000 |
| organamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0138 | -0.0088 | 0.0102 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0087 | 0.0035 | 0.0000 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0020 | -0.0015 | 0.0018 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0058 | -0.0082 | 0.0028 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0507 | 0.0475 | 0.0020 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0140 | 0.0172 | 0.0085 |
| organamnist / naive_async / iid | sync_fedavg / iid | -0.0043 | -0.0025 | 0.0070 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0070 | 0.0072 | 0.0055 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0512 | 0.0460 | 0.0000 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0097 | 0.0087 | 0.0042 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0098 | -0.0090 | 0.0060 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0003 | 0.0000 | 0.0055 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0440 | 0.0387 | 0.0000 |
| organcmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0115 | 0.0018 | 0.0132 |
| organcmnist / naive_async / iid | sync_fedavg / iid | 0.0063 | 0.0073 | 0.0010 |
| organcmnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0075 | -0.0072 | 0.0002 |
| pathmnist / agreement_fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0053 | 0.0253 | 0.0255 |
| pathmnist / agreement_fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0025 | 0.0167 | 0.0196 |
| pathmnist / fedbuff_async / dirichlet / inverse / B=5 | sync_fedavg / dirichlet | 0.0026 | 0.0157 | 0.0185 |
| pathmnist / staleness_async / dirichlet / inverse | sync_fedavg / dirichlet | 0.0242 | 0.0465 | 0.0277 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=3 | sync_fedavg / iid | 0.0403 | 0.7177 | 0.6844 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0286 | 0.0240 | 0.0024 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0252 | 0.0226 | 0.0043 |
| pathmnist / agreement_fedbuff_async / iid / polynomial / B=5 | sync_fedavg / iid | 0.0262 | 0.0244 | 0.0052 |
| pathmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0256 | 0.0202 | 0.0015 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0263 | 0.0234 | 0.0040 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0202 | 0.0228 | 0.0096 |
| pathmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0437 | 0.0414 | 0.0046 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=3 | sync_fedavg / iid | 0.4375 | 0.7177 | 0.2872 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0070 | 0.0040 | 0.0040 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0013 | 0.0029 | 0.0086 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0008 | 0.0021 | 0.0099 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0146 | 0.0114 | 0.0038 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0028 | -0.0001 | 0.0040 |
| pathmnist / agreement_fedbuff_async / iid / hinge / B=7 | sync_fedavg / iid | 0.0124 | 0.0113 | 0.0058 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0061 | 0.0033 | 0.0042 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0160 | 0.0118 | 0.0028 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0035 | -0.0061 | 0.0043 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0057 | -0.0006 | 0.0007 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0102 | 0.0079 | 0.0047 |
| pathmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0134 | 0.0117 | 0.0053 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0061 | 0.0079 | 0.0088 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0164 | 0.0149 | 0.0054 |
| pathmnist / naive_async / iid | sync_fedavg / iid | -0.0052 | -0.0025 | 0.0096 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0114 | 0.0114 | 0.0070 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0047 | 0.0050 | 0.0072 |
| pathmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0228 | 0.0252 | 0.0093 |
| pneumoniamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0176 | 0.0064 | 0.0240 |
| pneumoniamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0064 | -0.0032 | 0.0032 |
| pneumoniamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0208 | -0.0016 | 0.0192 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0016 | 0.0032 | 0.0048 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0064 | -0.0016 | 0.0048 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0064 | 0.0032 | 0.0096 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0128 | 0.0465 | 0.0337 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0032 | 0.0032 | 0.0000 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0048 | 0.0048 | 0.0000 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0048 | 0.0048 | 0.0000 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0609 | 0.0609 | 0.0000 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0032 | 0.0096 | 0.0064 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0064 | -0.0064 | 0.0000 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0016 | -0.0016 | 0.0000 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0417 | 0.0465 | 0.0048 |
| tissuemnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0058 | -0.0054 | 0.0004 |
| tissuemnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| tissuemnist / naive_async / iid | sync_fedavg / iid | -0.0022 | 0.0006 | 0.0028 |
| tissuemnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0020 | -0.0016 | 0.0004 |

- `Best Gap = sync_best_acc - async_best_acc`: peak accuracy cost of removing the barrier.
- `Final Gap = sync_final_acc - async_final_acc`: whether the async system actually converges near sync.
- `Stability Drop = best_acc - final_acc`: how much late-training regression or stale-update oscillation remains.

## Multi-Dataset Coverage

| Dataset | Task | Classes | Channels | Sync | Stateless Async | CAA-FedBuff |
|---|---|---:|---:|---:|---:|---:|
| bloodmnist | multi-class | 8 | 3 | 0.8825 | 0.8787 | 0.8857 |
| breastmnist | binary-class | 2 | 1 | 0.7308 | 0.7308 | 0.7308 |
| dermamnist | multi-class | 7 | 3 | 0.6928 | 0.6918 | 0.6898 |
| octmnist | multi-class | 4 | 1 | 0.5320 | 0.5320 | 0.5430 |
| organamnist | multi-class | 11 | 1 | 0.6265 | 0.6308 | 0.6402 |
| organcmnist | multi-class | 11 | 1 | 0.6428 | 0.6365 | 0.6542 |
| pathmnist | multi-class | 9 | 3 | 0.9110 | 0.9162 | 0.9145 |
| pneumoniamnist | binary-class | 2 | 1 | 0.8558 | 0.8526 | 0.8766 |
| tissuemnist | multi-class | 8 | 1 | 0.5564 | 0.5586 | 0.5622 |

## Stateless vs Staleness-Aware

This report treats `naive_async` as the stateless async baseline because it ignores logical staleness. `staleness_async` is the logical-staleness baseline.

| Dataset | Stateless Best | Staleness-Aware Best | CAA-Family Best | Note |
|---|---:|---:|---:|---|
| bloodmnist | 0.8787 | 0.8904 | 0.8857 | staleness decay helped; CAA matched/exceeded stateless |
| breastmnist | 0.7308 | 0.7308 | 0.7308 | staleness decay helped; CAA matched/exceeded stateless |
| dermamnist | 0.6918 | 0.6903 | 0.6898 | logical staleness alone was conservative |
| octmnist | 0.5320 | 0.5400 | 0.5430 | staleness decay helped; CAA matched/exceeded stateless |
| organamnist | 0.6308 | 0.6362 | 0.6402 | staleness decay helped; CAA matched/exceeded stateless |
| organcmnist | 0.6365 | 0.6502 | 0.6542 | staleness decay helped; CAA matched/exceeded stateless |
| pathmnist | 0.9162 | 0.9063 | 0.9145 | logical staleness alone was conservative |
| pneumoniamnist | 0.8526 | 0.8622 | 0.8766 | staleness decay helped; CAA matched/exceeded stateless |
| tissuemnist | 0.5586 | 0.5584 | 0.5622 | logical staleness alone was conservative; CAA matched/exceeded stateless |

## Interpretation

- Sync FedAvg is the stable baseline because the server waits for all clients.
- Naive async removes the waiting barrier, but stale updates can destabilize the trajectory.
- Staleness-aware async uses logical time to reduce the effect of old updates.
- FedBuff-lite buffers several asynchronous updates before applying them, connecting ML aggregation to distributed buffering and reordering trade-offs.
- CAA-FedBuff extends FedBuff-lite with update-direction agreement, delta-norm clipping, and adaptive server alpha while still avoiding a physical global clock.
- Dirichlet non-IID partitioning models hospitals with different patient/image distributions.
- Async-Sync gaps measure the price of removing the synchronization barrier: best gap for peak model quality, final gap for convergence, and stability drop for late-training regression.

## Figures

- `figures/test_acc_vs_progress.png`
- `figures/test_acc_vs_simulated_time.png`
- `figures/staleness_vs_event.png`
- `figures/effective_alpha_vs_event.png`
- `figures/agreement_vs_event.png`
- `figures/client_contribution_bar.png`
- `figures/report/best_accuracy_by_dataset.png`
- `figures/report/async_sync_best_gap_by_dataset.png`
- `figures/report/caa_gap_and_stability_by_dataset.png`
- `figures/report/stability_drop_by_dataset.png`
- `figures/report/accuracy_mean_std_by_dataset.png`
- `figures/report/async_sync_gap_errorbar.png`
- `figures/report/dataset_method_heatmap.png`
- `figures/report/existing_vs_ours_table.csv`
- `figures/classification/*_confusion.png`

## Future Extensions

- ChestMNIST and RetinaMNIST can be added later with task-specific loss and metrics; they are excluded from the current headline to keep the comparison fair.
- MobileNetV3-small can provide another compact edge-device backbone after the required small-CNN checks are complete.
- These are validation axes for future work, not headline claims for the current experiment set.
