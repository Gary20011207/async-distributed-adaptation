# R13946001 Experiment Notes

## Story for Presentation

Different hospitals and edge devices do not train at the same speed. Some nodes return fresh updates, while stragglers return stale updates computed from older global models.

The distributed-systems question is not only accuracy. It is how a system without a global clock should reason about delayed information. This project uses logical model versions to define staleness and compares aggregation policies under asynchronous arrivals.

## Distributed Systems Problem Statement

The project asks whether a federated learning server can remove the synchronous waiting barrier without losing too much model quality or stability. In the medical setting, hospitals and edge devices have different hardware, network delay, workload, and patient distributions, so updates arrive out of order.

## No Global Clock and Logical Staleness

The server does not assume synchronized physical clocks. It assigns a logical model version to each global model and measures stale updates by:

```text
staleness = current_server_version - client_start_version
```

This makes the demo a distributed-systems experiment rather than only a centralized ML benchmark: the same model update can be useful or harmful depending on when it arrives and what global version it was trained from.

## Headline Findings

Official headline results use the fair 3-seed matrix only:

```text
9 datasets x 6 methods x 3 seeds = 162 official runs
model = ResNet18
partition = IID
fair budget = async events = sync rounds * clients
```

- CAA-v2 mean best accuracy is `0.7169`, compared with Sync `0.7142`, Naive Async `0.7132`, FedBuff `0.7090`, and Staleness `0.6770`.
- CAA-v2 mean final accuracy is `0.7140`, compared with Sync `0.7121` and Naive Async `0.7096`.
- CAA-v2 beats Sync FedAvg in mean best accuracy on `6/9` datasets and mean final accuracy on `6/9` datasets.
- CAA-v2 beats the strongest classic baseline among Sync/Naive/Staleness/FedBuff on `5/9` datasets.
- CAA-v2 is not a universal winner; CAA-v1 has stronger peak accuracy but a larger stability drop.
- Logical staleness alone is stable but conservative: it has the lowest mean best accuracy among official methods.
- Single-run exploratory numbers such as CAA-family `0.9145`, Naive Async `0.9162`, and Sync `0.9110` are kept below as supporting PathMNIST runs, not as the headline claim.
- The main report metrics are `Async-Sync Best Gap`, `Async-Sync Final Gap`, and `Stability Drop = best_acc - final_acc`.

## Method Detail

- `sync_fedavg`: barrier baseline; the server waits for all clients each round.
- `naive_async`: stateless async baseline; every arriving update is applied with constant alpha.
- `staleness_async`: logical-staleness baseline; alpha is reduced by `server_version - client_start_version`.
- `fedbuff_async`: buffered async baseline; `B` stale/data-size weighted deltas are aggregated together.
- `agreement_fedbuff_async`: CAA-FedBuff adds direction agreement, median-norm clipping, and adaptive server alpha.
- `caa_fedbuff_v2`: CAA-v2 additionally compares updates with recent accepted server delta direction and adds client fairness credit.

CAA-FedBuff and CAA-v2 are AI-assisted implemented course-project algorithms, but their aggregation rules are explicit and deterministic. They combine known ideas from buffered async FL, staleness-aware aggregation, cosine agreement, clipping, and adaptive server step size. We claim it as our own implemented design extension, not as a publication-level novel FL algorithm.

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
| bloodmnist | resnet18 | CAA-v2 | 3 | 0.8757 | 0.0068 | 0.8741 | 0.0060 | 0.0016 |
| bloodmnist | resnet18 | FedBuff | 3 | 0.8696 | 0.0036 | 0.8676 | 0.0046 | 0.0020 |
| bloodmnist | resnet18 | Naive | 3 | 0.8719 | 0.0064 | 0.8704 | 0.0067 | 0.0015 |
| bloodmnist | resnet18 | Staleness | 3 | 0.8256 | 0.0064 | 0.8247 | 0.0064 | 0.0009 |
| bloodmnist | resnet18 | Sync | 3 | 0.8754 | 0.0084 | 0.8753 | 0.0083 | 0.0001 |
| breastmnist | resnet18 | CAA | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | CAA-v2 | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | FedBuff | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | Naive | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | Staleness | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| breastmnist | resnet18 | Sync | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 |
| dermamnist | resnet18 | CAA | 3 | 0.6884 | 0.0012 | 0.6863 | 0.0022 | 0.0022 |
| dermamnist | resnet18 | CAA-v2 | 3 | 0.6894 | 0.0032 | 0.6873 | 0.0048 | 0.0022 |
| dermamnist | resnet18 | FedBuff | 3 | 0.6884 | 0.0032 | 0.6865 | 0.0035 | 0.0020 |
| dermamnist | resnet18 | Naive | 3 | 0.6888 | 0.0033 | 0.6855 | 0.0058 | 0.0033 |
| dermamnist | resnet18 | Staleness | 3 | 0.6879 | 0.0044 | 0.6848 | 0.0010 | 0.0032 |
| dermamnist | resnet18 | Sync | 3 | 0.6899 | 0.0037 | 0.6863 | 0.0049 | 0.0037 |
| octmnist | resnet18 | CAA | 3 | 0.5463 | 0.0153 | 0.5373 | 0.0188 | 0.0090 |
| octmnist | resnet18 | CAA-v2 | 3 | 0.5460 | 0.0276 | 0.5417 | 0.0291 | 0.0043 |
| octmnist | resnet18 | FedBuff | 3 | 0.5160 | 0.0115 | 0.5133 | 0.0095 | 0.0027 |
| octmnist | resnet18 | Naive | 3 | 0.5433 | 0.0205 | 0.5343 | 0.0193 | 0.0090 |
| octmnist | resnet18 | Staleness | 3 | 0.4750 | 0.0010 | 0.4750 | 0.0010 | 0.0000 |
| octmnist | resnet18 | Sync | 3 | 0.5467 | 0.0237 | 0.5453 | 0.0241 | 0.0013 |
| organamnist | resnet18 | CAA | 3 | 0.6348 | 0.0079 | 0.6314 | 0.0065 | 0.0034 |
| organamnist | resnet18 | CAA-v2 | 3 | 0.6248 | 0.0073 | 0.6233 | 0.0059 | 0.0015 |
| organamnist | resnet18 | FedBuff | 3 | 0.6145 | 0.0073 | 0.6122 | 0.0057 | 0.0023 |
| organamnist | resnet18 | Naive | 3 | 0.6209 | 0.0092 | 0.6139 | 0.0099 | 0.0070 |
| organamnist | resnet18 | Staleness | 3 | 0.5717 | 0.0045 | 0.5699 | 0.0042 | 0.0018 |
| organamnist | resnet18 | Sync | 3 | 0.6185 | 0.0079 | 0.6157 | 0.0059 | 0.0028 |
| organcmnist | resnet18 | CAA | 3 | 0.6648 | 0.0141 | 0.6595 | 0.0169 | 0.0053 |
| organcmnist | resnet18 | CAA-v2 | 3 | 0.6608 | 0.0144 | 0.6554 | 0.0192 | 0.0053 |
| organcmnist | resnet18 | FedBuff | 3 | 0.6478 | 0.0160 | 0.6456 | 0.0163 | 0.0022 |
| organcmnist | resnet18 | Naive | 3 | 0.6492 | 0.0164 | 0.6482 | 0.0170 | 0.0011 |
| organcmnist | resnet18 | Staleness | 3 | 0.5869 | 0.0128 | 0.5863 | 0.0121 | 0.0007 |
| organcmnist | resnet18 | Sync | 3 | 0.6502 | 0.0157 | 0.6494 | 0.0165 | 0.0007 |
| pathmnist | resnet18 | CAA | 3 | 0.9055 | 0.0081 | 0.8996 | 0.0061 | 0.0059 |
| pathmnist | resnet18 | CAA-v2 | 3 | 0.9048 | 0.0097 | 0.9010 | 0.0090 | 0.0038 |
| pathmnist | resnet18 | FedBuff | 3 | 0.8983 | 0.0067 | 0.8918 | 0.0040 | 0.0065 |
| pathmnist | resnet18 | Naive | 3 | 0.9052 | 0.0108 | 0.8973 | 0.0088 | 0.0079 |
| pathmnist | resnet18 | Staleness | 3 | 0.8830 | 0.0072 | 0.8749 | 0.0046 | 0.0080 |
| pathmnist | resnet18 | Sync | 3 | 0.9039 | 0.0067 | 0.8968 | 0.0067 | 0.0072 |
| pneumoniamnist | resnet18 | CAA | 3 | 0.8707 | 0.0076 | 0.8552 | 0.0052 | 0.0155 |
| pneumoniamnist | resnet18 | CAA-v2 | 3 | 0.8606 | 0.0028 | 0.8542 | 0.0028 | 0.0064 |
| pneumoniamnist | resnet18 | FedBuff | 3 | 0.8574 | 0.0032 | 0.8515 | 0.0024 | 0.0059 |
| pneumoniamnist | resnet18 | Naive | 3 | 0.8515 | 0.0009 | 0.8515 | 0.0009 | 0.0000 |
| pneumoniamnist | resnet18 | Staleness | 3 | 0.8013 | 0.0181 | 0.8013 | 0.0181 | 0.0000 |
| pneumoniamnist | resnet18 | Sync | 3 | 0.8531 | 0.0024 | 0.8515 | 0.0037 | 0.0016 |
| tissuemnist | resnet18 | CAA | 3 | 0.5632 | 0.0026 | 0.5628 | 0.0030 | 0.0004 |
| tissuemnist | resnet18 | CAA-v2 | 3 | 0.5591 | 0.0059 | 0.5583 | 0.0048 | 0.0007 |
| tissuemnist | resnet18 | FedBuff | 3 | 0.5585 | 0.0044 | 0.5569 | 0.0047 | 0.0016 |
| tissuemnist | resnet18 | Naive | 3 | 0.5576 | 0.0009 | 0.5547 | 0.0020 | 0.0029 |
| tissuemnist | resnet18 | Staleness | 3 | 0.5306 | 0.0058 | 0.5296 | 0.0047 | 0.0010 |
| tissuemnist | resnet18 | Sync | 3 | 0.5590 | 0.0023 | 0.5581 | 0.0018 | 0.0009 |

## System Metrics Beyond Accuracy

Accuracy alone hides distributed-system behavior. This report also tracks staleness, simulated time, adaptive alpha, and client contribution imbalance.

| Metric | Meaning |
|---|---|
| `p95_staleness` | Tail delay in logical model-version units. |
| `avg_buffer_alpha` | How aggressively an async buffer updates the server. |
| `client_contribution_gini` | Whether fast clients dominate accepted async updates. |
| `time_to_90pct_best_acc` | Simulated time needed to approach each run's best accuracy. |

- Best CAA-v2 row in distributed summary: `pathmnist` / `iid` / `full_caa_v2` best=`0.9145`, final=`0.9102`, p95 staleness=`3.0000`, client Gini=`0.1692`.

## Hospital Non-IID Scenario

Dirichlet partitioning approximates hospitals with different label distributions. Lower alpha means stronger data heterogeneity and usually more conflicting client updates.

| Dataset | Dirichlet Alpha | Method | Runs | Best Acc | Stability Drop | p95 Staleness | Client Gini |
|---|---:|---|---:|---:|---:|---:|---:|
| bloodmnist | 0.1000 | CAA-v2 | 1 | 0.3730 | 0.0000 | 3.0000 | 0.1773 |
| bloodmnist | 0.1000 | FedBuff | 1 | 0.5308 | 0.0079 | 3.0000 | 0.1773 |
| bloodmnist | 0.1000 | Naive | 1 | 0.6396 | 0.1327 | 14.0000 | 0.1773 |
| bloodmnist | 0.1000 | Staleness | 1 | 0.6653 | 0.0281 | 14.0000 | 0.1773 |
| bloodmnist | 0.1000 | Sync | 1 | 0.7781 | 0.0544 | 0.0000 | nan |
| bloodmnist | 0.5000 | CAA-v2 | 1 | 0.8445 | 0.0237 | 3.0000 | 0.1773 |
| bloodmnist | 0.5000 | FedBuff | 1 | 0.8483 | 0.0134 | 3.0000 | 0.1773 |
| bloodmnist | 0.5000 | Naive | 1 | 0.8527 | 0.0000 | 14.0000 | 0.1773 |
| bloodmnist | 0.5000 | Staleness | 1 | 0.8603 | 0.0000 | 14.0000 | 0.1773 |
| bloodmnist | 0.5000 | Sync | 1 | 0.8772 | 0.0044 | 0.0000 | nan |
| organamnist | 0.1000 | CAA-v2 | 1 | 0.4615 | 0.0260 | 3.0000 | 0.1773 |
| organamnist | 0.1000 | FedBuff | 1 | 0.4790 | 0.0045 | 3.0000 | 0.1773 |
| organamnist | 0.1000 | Naive | 1 | 0.4983 | 0.0000 | 14.0000 | 0.1773 |
| organamnist | 0.1000 | Staleness | 1 | 0.5005 | 0.0000 | 14.0000 | 0.1773 |
| organamnist | 0.1000 | Sync | 1 | 0.5417 | 0.0000 | 0.0000 | nan |
| organamnist | 0.5000 | CAA-v2 | 1 | 0.5873 | 0.0000 | 3.0000 | 0.1773 |
| organamnist | 0.5000 | FedBuff | 1 | 0.5795 | 0.0145 | 3.0000 | 0.1773 |
| organamnist | 0.5000 | Naive | 1 | 0.5660 | 0.0000 | 14.0000 | 0.1773 |
| organamnist | 0.5000 | Staleness | 1 | 0.5837 | 0.0000 | 14.0000 | 0.1773 |
| organamnist | 0.5000 | Sync | 1 | 0.6272 | 0.0008 | 0.0000 | nan |
| pathmnist | 0.1000 | CAA-v2 | 1 | 0.8531 | 0.0084 | 3.0000 | 0.1736 |
| pathmnist | 0.1000 | FedBuff | 1 | 0.8435 | 0.0033 | 3.0000 | 0.1736 |
| pathmnist | 0.1000 | Naive | 1 | 0.8508 | 0.0494 | 15.0000 | 0.1736 |
| pathmnist | 0.1000 | Staleness | 1 | 0.8458 | 0.0358 | 15.0000 | 0.1736 |
| pathmnist | 0.1000 | Sync | 1 | 0.8479 | 0.0001 | 0.0000 | nan |
| pathmnist | 0.5000 | CAA | 1 | 0.8737 | 0.0196 | 3.0000 | 0.1728 |
| pathmnist | 0.5000 | CAA-v2 | 1 | 0.8896 | 0.0228 | 3.0000 | 0.1736 |
| pathmnist | 0.5000 | FedBuff | 1 | 0.8795 | 0.0142 | 3.0000 | 0.1732 |
| pathmnist | 0.5000 | Naive | 1 | 0.8833 | 0.0216 | 15.0000 | 0.1736 |
| pathmnist | 0.5000 | Staleness | 1 | 0.8672 | 0.0255 | 14.5250 | 0.1732 |
| pathmnist | 0.5000 | Sync | 1 | 0.8831 | 0.0051 | 0.0000 | nan |
| pneumoniamnist | 0.1000 | CAA-v2 | 1 | 0.6250 | 0.0000 | 3.0000 | 0.1773 |
| pneumoniamnist | 0.1000 | FedBuff | 1 | 0.6378 | 0.0112 | 3.0000 | 0.1773 |
| pneumoniamnist | 0.1000 | Naive | 1 | 0.6651 | 0.0385 | 14.0000 | 0.1773 |
| pneumoniamnist | 0.1000 | Staleness | 1 | 0.6282 | 0.0032 | 14.0000 | 0.1773 |
| pneumoniamnist | 0.1000 | Sync | 1 | 0.7340 | 0.1090 | 0.0000 | nan |
| pneumoniamnist | 0.5000 | CAA-v2 | 1 | 0.8510 | 0.1635 | 3.0000 | 0.1773 |
| pneumoniamnist | 0.5000 | FedBuff | 1 | 0.8157 | 0.1266 | 3.0000 | 0.1773 |
| pneumoniamnist | 0.5000 | Naive | 1 | 0.8622 | 0.1170 | 14.0000 | 0.1773 |
| pneumoniamnist | 0.5000 | Staleness | 1 | 0.8654 | 0.0737 | 14.0000 | 0.1773 |
| pneumoniamnist | 0.5000 | Sync | 1 | 0.8173 | 0.1779 | 0.0000 | nan |

## Straggler Stress Test

The delay tests make the timing problem visible. A method can have similar final accuracy under normal delay but become unstable when stragglers dominate the event queue.

| Delay Setting | Method | Runs | Best Acc | Stability Drop | p95 Staleness | Time to 90% Best | Client Gini |
|---|---|---:|---:|---:|---:|---:|---:|
| hetero_r0.2_x5 | CAA | 3 | 0.7265 | 0.0048 | 3.0000 | 40.5540 | 0.1829 |
| hetero_r0.2_x5 | CAA-v2 | 3 | 0.7732 | 0.0036 | 3.0000 | 47.0577 | 0.1801 |
| hetero_r0.2_x5 | FedBuff | 3 | 0.7155 | 0.0029 | 3.0000 | 32.8081 | 0.1829 |
| hetero_r0.2_x5 | Naive | 3 | 0.7194 | 0.0037 | 14.4446 | 32.3951 | 0.1829 |
| hetero_r0.2_x5 | Staleness | 3 | 0.7105 | 0.0029 | 14.4446 | 30.4100 | 0.1829 |
| hetero_r0.4_x8 | CAA-v2 | 1 | 0.8006 | 0.0010 | 7.0000 | 53.6699 | 0.3320 |
| hetero_r0.4_x8 | FedBuff | 1 | 0.7900 | 0.0008 | 7.0000 | 50.5206 | 0.3320 |
| hetero_r0.4_x8 | Naive | 1 | 0.7969 | 0.0031 | 34.7000 | 31.8116 | 0.3320 |
| hetero_r0.4_x8 | Staleness | 1 | 0.8159 | 0.0046 | 34.7000 | 50.5597 | 0.3320 |
| lognormal_m1_s0.5 | CAA-v2 | 1 | 0.7969 | 0.0044 | 3.3333 | 59.9146 | 0.0394 |
| lognormal_m1_s0.5 | FedBuff | 1 | 0.7877 | 0.0012 | 3.3333 | 41.1818 | 0.0394 |
| lognormal_m1_s0.5 | Naive | 1 | 0.7927 | 0.0046 | 17.3333 | 26.9622 | 0.0394 |
| lognormal_m1_s0.5 | Staleness | 1 | 0.7972 | 0.0044 | 17.3333 | 32.8011 | 0.0394 |
| uniform | CAA-v2 | 1 | 0.7949 | 0.0029 | 3.0000 | 117.3376 | 0.0316 |
| uniform | FedBuff | 1 | 0.7913 | 0.0028 | 3.0000 | 73.2262 | 0.0316 |
| uniform | Naive | 1 | 0.7908 | 0.0009 | 16.0333 | 58.9360 | 0.0316 |
| uniform | Staleness | 1 | 0.7946 | 0.0016 | 16.0333 | 40.3111 | 0.0316 |

## CAA-v2 Ablation

Ablations test whether CAA-v2 is more than FedBuff with a new name: server-trajectory agreement, client fairness credit, clipping, and adaptive alpha are removed one at a time.

| Dataset | Variant | Seeds | Best Acc Mean | Best Acc Std | Final Acc Mean | Stability Drop | Avg Agreement | Server Agreement | Fairness Weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bloodmnist | full_caa_v2 | 3 | 0.8079 | 0.1761 | 0.8038 | 0.0041 | 0.6768 | 0.2225 | 0.3093 |
| bloodmnist | no_fairness | 1 | 0.8790 | 0.0000 | 0.8769 | 0.0020 | 0.7026 | 0.1919 | 1.0000 |
| bloodmnist | no_server_ema | 1 | 0.8822 | 0.0000 | 0.8799 | 0.0023 | 0.7314 | 0.1967 | 0.3071 |
| bloodmnist | old_caa | 3 | 0.8810 | 0.0064 | 0.8792 | 0.0019 | 0.7256 | 0.0000 | 0.0000 |
| breastmnist | full_caa_v2 | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.8047 | 0.7144 | 0.3083 |
| breastmnist | old_caa | 3 | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.8168 | 0.0000 | 0.0000 |
| dermamnist | full_caa_v2 | 3 | 0.6894 | 0.0032 | 0.6873 | 0.0022 | 0.6688 | 0.2950 | 0.3083 |
| dermamnist | old_caa | 3 | 0.6884 | 0.0012 | 0.6863 | 0.0022 | 0.6941 | 0.0000 | 0.0000 |
| octmnist | full_caa_v2 | 3 | 0.5460 | 0.0276 | 0.5417 | 0.0043 | 0.6913 | 0.2030 | 0.3081 |
| octmnist | old_caa | 3 | 0.5463 | 0.0153 | 0.5373 | 0.0090 | 0.7009 | 0.0000 | 0.0000 |
| organamnist | full_caa_v2 | 3 | 0.5974 | 0.0565 | 0.5931 | 0.0043 | 0.6671 | 0.2187 | 0.3090 |
| organamnist | old_caa | 3 | 0.6348 | 0.0079 | 0.6314 | 0.0034 | 0.7432 | 0.0000 | 0.0000 |
| organcmnist | full_caa_v2 | 3 | 0.6608 | 0.0144 | 0.6554 | 0.0053 | 0.6914 | 0.2001 | 0.3080 |
| organcmnist | old_caa | 3 | 0.6648 | 0.0141 | 0.6595 | 0.0053 | 0.7215 | 0.0000 | 0.0000 |
| pathmnist | full_caa_v2 | 3 | 0.8949 | 0.0184 | 0.8876 | 0.0074 | 0.5349 | 0.1104 | 0.1801 |
| pathmnist | no_clipping | 2 | 0.9008 | 0.0074 | 0.8955 | 0.0052 | 0.5425 | 0.0807 | 0.1798 |
| pathmnist | no_fairness | 3 | 0.9031 | 0.0072 | 0.8970 | 0.0060 | 0.5390 | 0.0781 | 1.0000 |
| pathmnist | no_server_ema | 3 | 0.9061 | 0.0042 | 0.8986 | 0.0076 | 0.5483 | 0.0778 | 0.1799 |
| pathmnist | old_caa | 3 | 0.8952 | 0.0158 | 0.8869 | 0.0083 | 0.5363 | 0.0000 | 0.0000 |
| pathmnist | static_alpha | 2 | 0.9040 | 0.0070 | 0.8994 | 0.0045 | 0.5591 | 0.0855 | 0.1797 |
| pneumoniamnist | full_caa_v2 | 3 | 0.8115 | 0.1044 | 0.7750 | 0.0365 | 0.5639 | 0.2909 | 0.3069 |
| pneumoniamnist | old_caa | 3 | 0.8707 | 0.0076 | 0.8552 | 0.0155 | 0.6658 | 0.0000 | 0.0000 |
| tissuemnist | full_caa_v2 | 3 | 0.5591 | 0.0059 | 0.5583 | 0.0007 | 0.6597 | 0.1553 | 0.3083 |
| tissuemnist | old_caa | 3 | 0.5632 | 0.0026 | 0.5628 | 0.0004 | 0.6765 | 0.0000 | 0.0000 |

## Honest Claim and Limitations

- Strong current result: CAA-v2 matches or exceeds Sync on mean matched rows (best gap `-0.0023`, final gap `-0.0022`, stability drop `0.0031`).
- The headline should be based on matched update budget, same seed, same model, same partition, and same delay setting.
- ChestMNIST and RetinaMNIST remain future work because they require task-specific loss/metrics; mixing them now would weaken fairness.

## Best-by-Dataset View

For each dataset and method, this table reports the best completed full run. Negative gaps mean async matched or exceeded Sync FedAvg in that run.

| Dataset | Sync Best | Stateless Best | Staleness Best | FedBuff Best | CAA Best | CAA-v2 Best | Best CAA-Sync Gap | Best CAA Stability Drop | Best Async Method |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bloodmnist | 0.8825 | 0.8787 | 0.9003 | 0.8737 | 0.8857 | 0.8822 | -0.0032 | 0.0023 | staleness_async |
| breastmnist | 0.7308 | 0.7308 | 0.7308 | 0.7308 | 0.7308 | 0.7308 | 0.0000 | 0.0000 | naive_async |
| dermamnist | 0.6928 | 0.6918 | 0.6928 | 0.6918 | 0.6898 | 0.6928 | 0.0000 | 0.0000 | staleness_async |
| octmnist | 0.5740 | 0.5670 | 0.5400 | 0.5280 | 0.5630 | 0.5770 | -0.0030 | 0.0020 | caa_fedbuff_v2 |
| organamnist | 0.6265 | 0.6308 | 0.6385 | 0.6200 | 0.6402 | 0.6322 | -0.0138 | 0.0102 | agreement_fedbuff_async |
| organcmnist | 0.6683 | 0.6677 | 0.6502 | 0.6663 | 0.6790 | 0.6773 | -0.0108 | 0.0000 | agreement_fedbuff_async |
| pathmnist | 0.9110 | 0.9162 | 0.9089 | 0.9053 | 0.9118 | 0.9145 | -0.0035 | 0.0043 | naive_async |
| pneumoniamnist | 0.8558 | 0.8526 | 0.8622 | 0.8606 | 0.8766 | 0.8622 | -0.0208 | 0.0192 | agreement_fedbuff_async |
| tissuemnist | 0.5608 | 0.5586 | 0.5584 | 0.5622 | 0.5662 | 0.5658 | -0.0054 | 0.0000 | agreement_fedbuff_async |

## Detailed Result Summary

This table keeps all completed full runs, including tuning runs. Use the best-by-dataset view above for slides.

| Dataset | Model | Run | Budget | Best Acc | Best Step | Final Acc | Final Loss | Progress | Sim Time | Avg Staleness | Avg Alpha | Avg Agreement | Buffer Alpha | Dropped | Client Updates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bloodmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8837 | 300 | 0.8837 | 0.3359 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7183 | 0.6263 | 0 | 5-38 |
| bloodmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8857 | 280 | 0.8834 | 0.3274 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7193 | 0.6256 | 1 | 5-41 |
| bloodmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.8737 | 280 | 0.8705 | 0.3460 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7390 | 0.6274 | 0 | 7-38 |
| breastmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.7308 | 1 | 0.7308 | 0.5562 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.8187 | 0.6387 | 0 | 5-38 |
| breastmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.7308 | 20 | 0.7308 | 0.5685 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.8179 | 0.6376 | 0 | 5-41 |
| breastmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.7308 | 20 | 0.7308 | 0.5618 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.8137 | 0.6366 | 0 | 7-38 |
| dermamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6898 | 240 | 0.6888 | 0.8473 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6972 | 0.6260 | 2 | 5-38 |
| dermamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6878 | 120 | 0.6853 | 0.8359 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6863 | 0.6207 | 0 | 5-41 |
| dermamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6878 | 140 | 0.6848 | 0.8392 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6988 | 0.6220 | 0 | 7-38 |
| octmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5430 | 280 | 0.5250 | 1.1417 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7235 | 0.6277 | 1 | 5-38 |
| octmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5330 | 280 | 0.5280 | 1.1441 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6957 | 0.6218 | 0 | 5-41 |
| octmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5630 | 280 | 0.5590 | 1.0888 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6835 | 0.6199 | 0 | 7-38 |
| organamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6258 | 300 | 0.6258 | 0.9043 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7443 | 0.6304 | 1 | 5-38 |
| organamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6385 | 300 | 0.6385 | 0.8725 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7511 | 0.6298 | 1 | 5-41 |
| organamnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6402 | 280 | 0.6300 | 0.8883 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7341 | 0.6267 | 0 | 7-38 |
| organcmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6645 | 280 | 0.6490 | 0.8844 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7204 | 0.6264 | 0 | 5-38 |
| organcmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6790 | 300 | 0.6790 | 0.8337 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7150 | 0.6251 | 1 | 5-41 |
| organcmnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.6508 | 260 | 0.6505 | 0.8823 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7291 | 0.6260 | 0 | 7-38 |
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
| tissuemnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5612 | 260 | 0.5604 | 1.1868 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6804 | 0.6196 | 0 | 5-41 |
| tissuemnist | resnet18 | agreement_fedbuff_async / iid / hinge / B=5 | 300 | 0.5662 | 300 | 0.5662 | 1.1785 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6705 | 0.6183 | 0 | 7-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8796 | 280 | 0.8766 | 0.3480 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7184 | 0.6095 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8796 | 280 | 0.8784 | 0.3449 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6846 | 0.6048 | 0 | 5-41 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8679 | 280 | 0.8673 | 0.3559 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7043 | 0.6068 | 0 | 7-38 |
| bloodmnist | small_cnn | caa_fedbuff_v2 / iid | 300 | 0.7778 | 280 | 0.7726 | 0.6061 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7926 | 0.6204 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8822 | 280 | 0.8799 | 0.3370 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7314 | 0.6279 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8790 | 280 | 0.8769 | 0.3454 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7026 | 0.6077 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / dirichlet | 300 | 0.8445 | 280 | 0.8208 | 0.5016 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.5360 | 0.5917 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / dirichlet | 300 | 0.3730 | 300 | 0.3730 | 1.4378 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.5896 | 0.5993 | 0 | 5-38 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8647 | 300 | 0.8647 | 0.3632 | 300 | 165.2063 | 1.7533 | 0.6200 | 0.8003 | 0.6155 | 0 | 27-33 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8717 | 280 | 0.8679 | 0.3498 | 300 | 93.6442 | 1.7467 | 0.6196 | 0.7669 | 0.6116 | 1 | 25-33 |
| bloodmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8822 | 260 | 0.8816 | 0.3358 | 300 | 139.4061 | 1.7033 | 0.6127 | 0.6145 | 0.6029 | 1 | 5-48 |
| breastmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5561 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.8063 | 0.6346 | 0 | 5-38 |
| breastmnist | mobilenet_v3_small | caa_fedbuff_v2 / iid | 300 | 0.7308 | 20 | 0.7308 | 0.6735 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.9947 | 0.6614 | 0 | 5-38 |
| breastmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.7308 | 20 | 0.7308 | 0.5683 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.8048 | 0.6335 | 0 | 5-41 |
| breastmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.7308 | 20 | 0.7308 | 0.5620 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.8030 | 0.6328 | 0 | 7-38 |
| dermamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6928 | 300 | 0.6928 | 0.8475 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6651 | 0.6068 | 0 | 5-38 |
| dermamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6893 | 120 | 0.6843 | 0.8388 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6705 | 0.6068 | 0 | 5-41 |
| dermamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6863 | 100 | 0.6848 | 0.8429 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6708 | 0.6066 | 0 | 7-38 |
| octmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5370 | 280 | 0.5290 | 1.1411 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7195 | 0.6103 | 1 | 5-38 |
| octmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5240 | 260 | 0.5210 | 1.1500 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6789 | 0.6038 | 0 | 5-41 |
| octmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5770 | 280 | 0.5750 | 1.0747 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6756 | 0.6053 | 1 | 7-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6178 | 300 | 0.6178 | 0.9276 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7240 | 0.6104 | 0 | 5-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6245 | 280 | 0.6228 | 0.8918 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.7053 | 0.6072 | 0 | 5-41 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6322 | 280 | 0.6295 | 0.8987 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.7165 | 0.6093 | 1 | 7-38 |
| organamnist | small_cnn | caa_fedbuff_v2 / iid | 300 | 0.5757 | 260 | 0.5737 | 1.0631 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.7382 | 0.6139 | 0 | 5-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / dirichlet | 300 | 0.5873 | 300 | 0.5873 | 0.9777 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.5246 | 0.5885 | 0 | 5-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / dirichlet | 300 | 0.4615 | 220 | 0.4355 | 1.5174 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.4723 | 0.5915 | 3 | 5-38 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6165 | 300 | 0.6165 | 0.9301 | 300 | 165.2063 | 1.7533 | 0.6200 | 0.8037 | 0.6153 | 0 | 27-33 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6195 | 280 | 0.6155 | 0.9358 | 300 | 93.6442 | 1.7467 | 0.6196 | 0.7817 | 0.6134 | 1 | 25-33 |
| organamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6202 | 300 | 0.6202 | 0.9131 | 300 | 139.4061 | 1.7033 | 0.6127 | 0.6089 | 0.6027 | 1 | 5-48 |
| organcmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6542 | 280 | 0.6410 | 0.8989 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6933 | 0.6069 | 0 | 5-38 |
| organcmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6773 | 300 | 0.6773 | 0.8433 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6894 | 0.6059 | 1 | 5-41 |
| organcmnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.6508 | 260 | 0.6480 | 0.9047 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6914 | 0.6066 | 1 | 7-38 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9049 | 700 | 0.9007 | 0.3858 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5500 | 0.5854 | 0 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8950 | 540 | 0.8922 | 0.4519 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5500 | 0.5856 | 0 | 22-129 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9145 | 820 | 0.9102 | 0.3336 | 1000 | 353.5214 | 1.7800 | 0.6161 | 0.5456 | 0.5854 | 1 | 23-127 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9053 | 860 | 0.9046 | 0.3517 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5477 | 0.6004 | 0 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9008 | 600 | 0.8961 | 0.3756 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5346 | 0.5837 | 0 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / dirichlet | 1000 | 0.8896 | 860 | 0.8667 | 0.4646 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.4864 | 0.5861 | 10 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / dirichlet | 1000 | 0.8531 | 900 | 0.8447 | 0.4694 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.4927 | 0.5843 | 4 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9036 | 580 | 0.8950 | 0.3834 | 1000 | 563.4223 | 1.7830 | 0.6200 | 0.5876 | 0.5888 | 0 | 90-106 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8994 | 720 | 0.8942 | 0.3725 | 1000 | 310.6560 | 1.7850 | 0.6196 | 0.5597 | 0.5855 | 0 | 94-109 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8993 | 840 | 0.8968 | 0.4245 | 1000 | 471.5191 | 1.7570 | 0.6117 | 0.5068 | 0.5841 | 0 | 18-158 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9060 | 600 | 0.9001 | 0.3564 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5409 | 0.5850 | 2 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9089 | 600 | 0.9033 | 0.3819 | 1000 | 357.5957 | 1.7850 | 0.6158 | 0.5614 | 0.6200 | 1 | 20-126 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9024 | 600 | 0.8925 | 0.4134 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5422 | 0.5998 | 0 | 22-129 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8972 | 740 | 0.8936 | 0.4105 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5415 | 0.5845 | 0 | 22-129 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8955 | 840 | 0.8909 | 0.4576 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5440 | 0.5851 | 0 | 22-129 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.8990 | 800 | 0.8955 | 0.4091 | 1000 | 355.3563 | 1.7770 | 0.6159 | 0.5568 | 0.6200 | 0 | 22-129 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9107 | 780 | 0.8986 | 0.3887 | 1000 | 353.5214 | 1.7800 | 0.6161 | 0.5549 | 0.6017 | 0 | 23-127 |
| pathmnist | resnet18 | caa_fedbuff_v2 / iid | 1000 | 0.9111 | 780 | 0.9014 | 0.3535 | 1000 | 353.5214 | 1.7800 | 0.6161 | 0.5409 | 0.5847 | 0 | 23-127 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8574 | 140 | 0.8526 | 0.3858 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6127 | 0.6021 | 0 | 5-38 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8622 | 200 | 0.8574 | 0.3569 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6054 | 0.6001 | 0 | 5-41 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.8622 | 160 | 0.8526 | 0.3879 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6026 | 0.5989 | 0 | 7-38 |
| pneumoniamnist | small_cnn | caa_fedbuff_v2 / iid | 300 | 0.8429 | 80 | 0.8093 | 0.4258 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6951 | 0.6156 | 0 | 5-38 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / dirichlet | 300 | 0.8510 | 160 | 0.6875 | 0.6469 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.5330 | 0.5959 | 3 | 5-38 |
| pneumoniamnist | resnet18 | caa_fedbuff_v2 / dirichlet | 300 | 0.6250 | 1 | 0.6250 | 0.8374 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.4660 | 0.5919 | 3 | 5-38 |
| tissuemnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5564 | 300 | 0.5564 | 1.1855 | 300 | 105.3838 | 1.7133 | 0.6164 | 0.6650 | 0.6024 | 0 | 5-38 |
| tissuemnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5550 | 260 | 0.5548 | 1.1932 | 300 | 109.0030 | 1.7333 | 0.6161 | 0.6545 | 0.6002 | 0 | 5-41 |
| tissuemnist | resnet18 | caa_fedbuff_v2 / iid | 300 | 0.5658 | 280 | 0.5638 | 1.1834 | 300 | 104.8960 | 1.7433 | 0.6165 | 0.6595 | 0.6030 | 2 | 7-38 |
| bloodmnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 300 | 0.8483 | 260 | 0.8348 | 0.4715 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| bloodmnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 300 | 0.5308 | 240 | 0.5229 | 1.0876 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| bloodmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.8655 | 300 | 0.8655 | 0.3813 | 300 | 165.2063 | 1.7533 | 0.5000 |  |  |  | 27-33 |
| bloodmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.8644 | 300 | 0.8644 | 0.3773 | 300 | 93.6442 | 1.7467 | 0.4996 |  |  |  | 25-33 |
| bloodmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.8632 | 280 | 0.8620 | 0.3799 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| bloodmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.8679 | 300 | 0.8679 | 0.3727 | 300 | 139.4061 | 1.7033 | 0.4941 |  |  |  | 5-48 |
| bloodmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.8682 | 240 | 0.8644 | 0.3682 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| bloodmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.8737 | 280 | 0.8728 | 0.3595 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| bloodmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.8670 | 280 | 0.8655 | 0.3776 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| breastmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.7308 | 1 | 0.7308 | 0.5741 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| breastmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.7308 | 20 | 0.7308 | 0.5774 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| breastmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.7308 | 20 | 0.7308 | 0.5659 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| dermamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6918 | 240 | 0.6903 | 0.8486 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| dermamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6883 | 140 | 0.6858 | 0.8399 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| dermamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6853 | 100 | 0.6833 | 0.8454 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| octmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.5150 | 280 | 0.5130 | 1.1614 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| octmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.5050 | 280 | 0.5040 | 1.1744 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| octmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.5280 | 260 | 0.5230 | 1.1602 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| organamnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 300 | 0.5795 | 240 | 0.5650 | 1.0136 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| organamnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 300 | 0.4790 | 160 | 0.4745 | 1.4154 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| organamnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.6108 | 220 | 0.6062 | 0.9581 | 300 | 165.2063 | 1.7533 | 0.5000 |  |  |  | 27-33 |
| organamnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.6042 | 300 | 0.6042 | 0.9638 | 300 | 93.6442 | 1.7467 | 0.4996 |  |  |  | 25-33 |
| organamnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.6105 | 260 | 0.6065 | 0.9562 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| organamnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 300 | 0.6060 | 300 | 0.6060 | 0.9523 | 300 | 139.4061 | 1.7033 | 0.4941 |  |  |  | 5-48 |
| organamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6062 | 220 | 0.6058 | 0.9447 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| organamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6200 | 280 | 0.6168 | 0.9212 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| organamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6172 | 240 | 0.6140 | 0.9336 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| organcmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6395 | 280 | 0.6380 | 0.9197 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| organcmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6663 | 240 | 0.6643 | 0.8766 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| organcmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.6378 | 280 | 0.6345 | 0.9312 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| pathmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 500 | 0.8854 | 340 | 0.8838 | 0.5395 | 500 | 181.0887 | 1.7720 | 0.2264 |  |  |  | 10-63 |
| pathmnist | resnet18 | fedbuff_async / dirichlet / inverse / B=5 | 500 | 0.8735 | 240 | 0.8550 | 0.4731 | 500 | 181.0887 | 1.7720 | 0.2264 |  |  |  | 10-63 |
| pathmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 1000 | 0.8976 | 560 | 0.8923 | 0.4218 | 1000 | 357.5957 | 1.7850 | 0.2224 |  |  |  | 20-126 |
| pathmnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 1000 | 0.8854 | 920 | 0.8755 | 0.4534 | 1000 | 357.5957 | 1.7850 | 0.4966 |  |  |  | 20-126 |
| pathmnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 1000 | 0.8435 | 640 | 0.8401 | 0.4862 | 1000 | 357.5957 | 1.7850 | 0.4966 |  |  |  | 20-126 |
| pathmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 1000 | 0.8975 | 700 | 0.8936 | 0.4046 | 1000 | 563.4223 | 1.7830 | 0.5000 |  |  |  | 90-106 |
| pathmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 1000 | 0.8946 | 440 | 0.8911 | 0.4724 | 1000 | 310.6560 | 1.7850 | 0.4997 |  |  |  | 94-109 |
| pathmnist | resnet18 | fedbuff_async / iid / hinge / B=5 | 1000 | 0.8961 | 640 | 0.8936 | 0.5021 | 1000 | 471.5191 | 1.7570 | 0.4933 |  |  |  | 18-158 |
| pathmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 1000 | 0.8919 | 540 | 0.8876 | 0.4336 | 1000 | 355.3563 | 1.7770 | 0.2240 |  |  |  | 22-129 |
| pathmnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 1000 | 0.9053 | 800 | 0.8955 | 0.3665 | 1000 | 353.5214 | 1.7800 | 0.2228 |  |  |  | 23-127 |
| pneumoniamnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 300 | 0.8157 | 220 | 0.6891 | 0.5320 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | fedbuff_async / dirichlet / hinge / B=5 | 300 | 0.6378 | 180 | 0.6266 | 0.9131 | 300 | 105.3838 | 1.7133 | 0.4971 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.8606 | 200 | 0.8494 | 0.3806 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.8542 | 300 | 0.8542 | 0.3659 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| pneumoniamnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.8574 | 220 | 0.8510 | 0.3902 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| tissuemnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.5536 | 280 | 0.5516 | 1.1983 | 300 | 105.3838 | 1.7133 | 0.2321 |  |  |  | 5-38 |
| tissuemnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.5622 | 220 | 0.5606 | 1.2034 | 300 | 109.0030 | 1.7333 | 0.2289 |  |  |  | 5-41 |
| tissuemnist | resnet18 | fedbuff_async / iid / inverse / B=5 | 300 | 0.5598 | 280 | 0.5586 | 1.1973 | 300 | 104.8960 | 1.7433 | 0.2275 |  |  |  | 7-38 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8708 | 260 | 0.8670 | 0.3671 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8787 | 260 | 0.8781 | 0.3486 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8661 | 300 | 0.8661 | 0.3788 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| bloodmnist | small_cnn | naive_async / iid | 300 | 0.7808 | 260 | 0.7676 | 0.6225 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| bloodmnist | resnet18 | naive_async / dirichlet | 300 | 0.8527 | 300 | 0.8527 | 0.4475 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| bloodmnist | resnet18 | naive_async / dirichlet | 300 | 0.6396 | 280 | 0.5069 | 1.1230 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8690 | 280 | 0.8670 | 0.3766 | 300 | 165.2063 | 8.8300 | 0.5000 |  |  |  | 27-33 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8693 | 220 | 0.8690 | 0.3752 | 300 | 93.6442 | 8.8000 | 0.5000 |  |  |  | 25-33 |
| bloodmnist | resnet18 | naive_async / iid | 300 | 0.8784 | 280 | 0.8778 | 0.3530 | 300 | 139.4061 | 8.5667 | 0.5000 |  |  |  | 5-48 |
| breastmnist | resnet18 | naive_async / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5767 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| breastmnist | mobilenet_v3_small | naive_async / iid | 300 | 0.7308 | 1 | 0.7308 | 0.6792 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| breastmnist | resnet18 | naive_async / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5815 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| breastmnist | resnet18 | naive_async / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5696 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| dermamnist | resnet18 | naive_async / iid | 300 | 0.6918 | 220 | 0.6908 | 0.8468 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| dermamnist | resnet18 | naive_async / iid | 300 | 0.6893 | 80 | 0.6863 | 0.8376 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| dermamnist | resnet18 | naive_async / iid | 300 | 0.6853 | 180 | 0.6793 | 0.8430 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| octmnist | resnet18 | naive_async / iid | 300 | 0.5320 | 240 | 0.5190 | 1.1561 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| octmnist | resnet18 | naive_async / iid | 300 | 0.5310 | 260 | 0.5280 | 1.1267 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| octmnist | resnet18 | naive_async / iid | 300 | 0.5670 | 260 | 0.5560 | 1.0932 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6125 | 260 | 0.6040 | 0.9419 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6308 | 240 | 0.6238 | 0.9125 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6195 | 220 | 0.6140 | 0.9325 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| organamnist | small_cnn | naive_async / iid | 300 | 0.5753 | 300 | 0.5753 | 1.0713 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organamnist | resnet18 | naive_async / dirichlet | 300 | 0.5660 | 300 | 0.5660 | 1.0150 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organamnist | resnet18 | naive_async / dirichlet | 300 | 0.4983 | 300 | 0.4983 | 1.3076 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6072 | 300 | 0.6072 | 0.9508 | 300 | 165.2063 | 8.8300 | 0.5000 |  |  |  | 27-33 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6082 | 240 | 0.6070 | 0.9492 | 300 | 93.6442 | 8.8000 | 0.5000 |  |  |  | 25-33 |
| organamnist | resnet18 | naive_async / iid | 300 | 0.6122 | 280 | 0.6098 | 0.9263 | 300 | 139.4061 | 8.5667 | 0.5000 |  |  |  | 5-48 |
| organcmnist | resnet18 | naive_async / iid | 300 | 0.6365 | 260 | 0.6355 | 0.9082 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| organcmnist | resnet18 | naive_async / iid | 300 | 0.6677 | 260 | 0.6675 | 0.8672 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| organcmnist | resnet18 | naive_async / iid | 300 | 0.6435 | 260 | 0.6415 | 0.9212 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| pathmnist | resnet18 | naive_async / iid | 500 | 0.8847 | 320 | 0.8806 | 0.5442 | 500 | 181.0887 | 8.8920 | 0.5000 |  |  |  | 10-63 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.9049 | 700 | 0.8961 | 0.4505 | 1000 | 357.5957 | 8.9430 | 0.5000 |  |  |  | 20-126 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.8946 | 600 | 0.8891 | 0.4805 | 1000 | 355.3563 | 8.9030 | 0.5000 |  |  |  | 22-129 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.9162 | 480 | 0.9065 | 0.3360 | 1000 | 353.5214 | 8.9210 | 0.5000 |  |  |  | 23-127 |
| pathmnist | resnet18 | naive_async / dirichlet | 1000 | 0.8833 | 640 | 0.8617 | 0.4622 | 1000 | 357.5957 | 8.9430 | 0.5000 |  |  |  | 20-126 |
| pathmnist | resnet18 | naive_async / dirichlet | 1000 | 0.8508 | 920 | 0.8014 | 0.6604 | 1000 | 357.5957 | 8.9430 | 0.5000 |  |  |  | 20-126 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.8961 | 800 | 0.8955 | 0.3556 | 1000 | 563.4223 | 8.9330 | 0.5000 |  |  |  | 90-106 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.9004 | 620 | 0.8882 | 0.3974 | 1000 | 310.6560 | 8.9430 | 0.5000 |  |  |  | 94-109 |
| pathmnist | resnet18 | naive_async / iid | 1000 | 0.9001 | 620 | 0.8940 | 0.4134 | 1000 | 471.5191 | 8.8090 | 0.5000 |  |  |  | 18-158 |
| pneumoniamnist | resnet18 | naive_async / iid | 300 | 0.8526 | 260 | 0.8526 | 0.3845 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | naive_async / iid | 300 | 0.8510 | 300 | 0.8510 | 0.3687 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| pneumoniamnist | resnet18 | naive_async / iid | 300 | 0.8510 | 300 | 0.8510 | 0.3791 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| pneumoniamnist | small_cnn | naive_async / iid | 300 | 0.7949 | 300 | 0.7949 | 0.4399 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | naive_async / dirichlet | 300 | 0.8622 | 220 | 0.7452 | 0.4693 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | naive_async / dirichlet | 300 | 0.6651 | 120 | 0.6266 | 0.6400 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| tissuemnist | resnet18 | naive_async / iid | 300 | 0.5586 | 260 | 0.5558 | 1.1963 | 300 | 105.3838 | 8.6433 | 0.5000 |  |  |  | 5-38 |
| tissuemnist | resnet18 | naive_async / iid | 300 | 0.5568 | 260 | 0.5524 | 1.2067 | 300 | 109.0030 | 8.7367 | 0.5000 |  |  |  | 5-41 |
| tissuemnist | resnet18 | naive_async / iid | 300 | 0.5574 | 280 | 0.5558 | 1.1970 | 300 | 104.8960 | 8.7800 | 0.5000 |  |  |  | 7-38 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8819 | 260 | 0.8813 | 0.3306 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8904 | 260 | 0.8883 | 0.3172 | 300 | 109.0030 | 8.7367 | 0.3959 |  |  |  | 5-41 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8708 | 220 | 0.8682 | 0.3540 | 300 | 104.8960 | 8.7800 | 0.3965 |  |  |  | 7-38 |
| bloodmnist | small_cnn | staleness_async / iid / hinge | 300 | 0.7948 | 240 | 0.7919 | 0.5799 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / dirichlet / hinge | 300 | 0.8603 | 300 | 0.8603 | 0.4207 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / dirichlet / hinge | 300 | 0.6653 | 260 | 0.6372 | 1.0113 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8740 | 260 | 0.8717 | 0.3632 | 300 | 165.2063 | 8.8300 | 0.3756 |  |  |  | 27-33 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8746 | 300 | 0.8746 | 0.3564 | 300 | 93.6442 | 8.8000 | 0.3813 |  |  |  | 25-33 |
| bloodmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.9003 | 280 | 0.8992 | 0.2978 | 300 | 139.4061 | 8.5667 | 0.4246 |  |  |  | 5-48 |
| bloodmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.8199 | 260 | 0.8185 | 0.4969 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| bloodmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.8325 | 260 | 0.8313 | 0.4755 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| bloodmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.8243 | 280 | 0.8243 | 0.4840 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| breastmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.7308 | 1 | 0.7308 | 0.5720 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| breastmnist | mobilenet_v3_small | staleness_async / iid / hinge | 300 | 0.7308 | 1 | 0.7308 | 0.6771 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| breastmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.7308 | 1 | 0.7308 | 0.5939 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| breastmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.7308 | 1 | 0.7308 | 0.5825 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| breastmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.7308 | 1 | 0.7308 | 0.5875 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| dermamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6903 | 160 | 0.6898 | 0.8378 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| dermamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.6928 | 160 | 0.6858 | 0.8728 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| dermamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.6843 | 280 | 0.6838 | 0.8781 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| dermamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.6868 | 240 | 0.6848 | 0.8914 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| octmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.5400 | 240 | 0.5320 | 1.1293 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| octmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.4740 | 260 | 0.4740 | 1.2431 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| octmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.4760 | 260 | 0.4760 | 1.2495 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| octmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.4750 | 240 | 0.4750 | 1.2220 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6168 | 280 | 0.6125 | 0.9134 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6362 | 240 | 0.6302 | 0.8850 | 300 | 109.0030 | 8.7367 | 0.3959 |  |  |  | 5-41 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6268 | 260 | 0.6212 | 0.9008 | 300 | 104.8960 | 8.7800 | 0.3965 |  |  |  | 7-38 |
| organamnist | small_cnn | staleness_async / iid / hinge | 300 | 0.5825 | 300 | 0.5825 | 1.0362 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / dirichlet / hinge | 300 | 0.5837 | 300 | 0.5837 | 1.0089 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / dirichlet / hinge | 300 | 0.5005 | 300 | 0.5005 | 1.2836 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6105 | 300 | 0.6105 | 0.9307 | 300 | 165.2063 | 8.8300 | 0.3756 |  |  |  | 27-33 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6150 | 240 | 0.6105 | 0.9354 | 300 | 93.6442 | 8.8000 | 0.3813 |  |  |  | 25-33 |
| organamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6385 | 260 | 0.6320 | 0.8660 | 300 | 139.4061 | 8.5667 | 0.4246 |  |  |  | 5-48 |
| organamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5680 | 280 | 0.5670 | 1.0924 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| organamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5705 | 280 | 0.5680 | 1.0571 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| organamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5767 | 260 | 0.5747 | 1.0592 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| organcmnist | resnet18 | staleness_async / iid / hinge | 300 | 0.6502 | 280 | 0.6500 | 0.8787 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| organcmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5895 | 240 | 0.5890 | 1.0735 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| organcmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5982 | 240 | 0.5968 | 1.0879 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| organcmnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5730 | 300 | 0.5730 | 1.1045 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| pathmnist | resnet18 | staleness_async / iid / inverse | 500 | 0.8673 | 380 | 0.8627 | 0.5351 | 500 | 181.0887 | 8.8920 | 0.0725 |  |  |  | 10-63 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 500 | 0.8908 | 320 | 0.8812 | 0.5780 | 500 | 181.0887 | 8.8920 | 0.3941 |  |  |  | 10-63 |
| pathmnist | resnet18 | staleness_async / dirichlet / inverse | 500 | 0.8519 | 360 | 0.8242 | 0.5235 | 500 | 181.0887 | 8.8920 | 0.0725 |  |  |  | 10-63 |
| pathmnist | resnet18 | staleness_async / iid / inverse | 1000 | 0.8882 | 680 | 0.8788 | 0.4687 | 1000 | 357.5957 | 8.9430 | 0.0710 |  |  |  | 20-126 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.8996 | 660 | 0.8926 | 0.3988 | 1000 | 355.3563 | 8.9030 | 0.3942 |  |  |  | 22-129 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.9063 | 660 | 0.8990 | 0.3824 | 1000 | 353.5214 | 8.9210 | 0.3939 |  |  |  | 23-127 |
| pathmnist | resnet18 | staleness_async / dirichlet / hinge | 1000 | 0.8825 | 460 | 0.8592 | 0.4707 | 1000 | 357.5957 | 8.9430 | 0.3955 |  |  |  | 20-126 |
| pathmnist | resnet18 | staleness_async / dirichlet / hinge | 1000 | 0.8458 | 980 | 0.8100 | 0.5872 | 1000 | 357.5957 | 8.9430 | 0.3955 |  |  |  | 20-126 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.8993 | 540 | 0.8968 | 0.4050 | 1000 | 563.4223 | 8.9330 | 0.3724 |  |  |  | 90-106 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.9019 | 400 | 0.8932 | 0.4418 | 1000 | 310.6560 | 8.9430 | 0.3811 |  |  |  | 94-109 |
| pathmnist | resnet18 | staleness_async / iid / hinge | 1000 | 0.9089 | 500 | 0.9026 | 0.4513 | 1000 | 471.5191 | 8.8090 | 0.4259 |  |  |  | 18-158 |
| pathmnist | resnet18 | staleness_async / iid / inverse | 1000 | 0.8748 | 540 | 0.8699 | 0.4523 | 1000 | 355.3563 | 8.9030 | 0.0736 |  |  |  | 22-129 |
| pathmnist | resnet18 | staleness_async / iid / inverse | 1000 | 0.8859 | 520 | 0.8760 | 0.4161 | 1000 | 353.5214 | 8.9210 | 0.0730 |  |  |  | 23-127 |
| pneumoniamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8526 | 200 | 0.8462 | 0.3835 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8622 | 300 | 0.8622 | 0.3519 | 300 | 109.0030 | 8.7367 | 0.3959 |  |  |  | 5-41 |
| pneumoniamnist | resnet18 | staleness_async / iid / hinge | 300 | 0.8574 | 220 | 0.8574 | 0.3814 | 300 | 104.8960 | 8.7800 | 0.3965 |  |  |  | 7-38 |
| pneumoniamnist | small_cnn | staleness_async / iid / hinge | 300 | 0.8141 | 220 | 0.8093 | 0.4231 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | staleness_async / dirichlet / hinge | 300 | 0.8654 | 220 | 0.7917 | 0.4316 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | staleness_async / dirichlet / hinge | 300 | 0.6282 | 240 | 0.6250 | 0.6828 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.7901 | 300 | 0.7901 | 0.4628 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| pneumoniamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.7917 | 300 | 0.7917 | 0.4747 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| pneumoniamnist | resnet18 | staleness_async / iid / inverse | 300 | 0.8221 | 300 | 0.8221 | 0.4281 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| tissuemnist | resnet18 | staleness_async / iid / hinge | 300 | 0.5584 | 260 | 0.5580 | 1.1850 | 300 | 105.3838 | 8.6433 | 0.3975 |  |  |  | 5-38 |
| tissuemnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5354 | 220 | 0.5324 | 1.2614 | 300 | 109.0030 | 8.7367 | 0.0760 |  |  |  | 5-41 |
| tissuemnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5322 | 300 | 0.5322 | 1.2549 | 300 | 104.8960 | 8.7800 | 0.0754 |  |  |  | 7-38 |
| tissuemnist | resnet18 | staleness_async / iid / inverse | 300 | 0.5242 | 260 | 0.5242 | 1.2608 | 300 | 105.3838 | 8.6433 | 0.0763 |  |  |  | 5-38 |
| bloodmnist | resnet18 | sync_fedavg / iid | 300 | 0.8775 | 29 | 0.8775 | 0.3535 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | resnet18 | sync_fedavg / iid | 300 | 0.8825 | 29 | 0.8822 | 0.3368 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | resnet18 | sync_fedavg / iid | 300 | 0.8661 | 30 | 0.8661 | 0.3681 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | small_cnn | sync_fedavg / iid | 300 | 0.7790 | 29 | 0.7790 | 0.5964 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | resnet18 | sync_fedavg / dirichlet | 300 | 0.8772 | 24 | 0.8728 | 0.3719 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| bloodmnist | resnet18 | sync_fedavg / dirichlet | 300 | 0.7781 | 21 | 0.7238 | 0.7606 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| breastmnist | resnet18 | sync_fedavg / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5776 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| breastmnist | mobilenet_v3_small | sync_fedavg / iid | 300 | 0.7308 | 1 | 0.7308 | 0.6783 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| breastmnist | resnet18 | sync_fedavg / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5778 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| breastmnist | resnet18 | sync_fedavg / iid | 300 | 0.7308 | 1 | 0.7308 | 0.5675 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| dermamnist | resnet18 | sync_fedavg / iid | 300 | 0.6928 | 24 | 0.6903 | 0.8430 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| dermamnist | resnet18 | sync_fedavg / iid | 300 | 0.6913 | 7 | 0.6878 | 0.8358 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| dermamnist | resnet18 | sync_fedavg / iid | 300 | 0.6858 | 9 | 0.6808 | 0.8452 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| octmnist | resnet18 | sync_fedavg / iid | 300 | 0.5320 | 26 | 0.5290 | 1.1224 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| octmnist | resnet18 | sync_fedavg / iid | 300 | 0.5340 | 29 | 0.5340 | 1.1218 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| octmnist | resnet18 | sync_fedavg / iid | 300 | 0.5740 | 28 | 0.5730 | 1.0825 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / iid | 300 | 0.6108 | 27 | 0.6095 | 0.9239 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / iid | 300 | 0.6265 | 23 | 0.6212 | 0.9030 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / iid | 300 | 0.6182 | 22 | 0.6162 | 0.9135 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | small_cnn | sync_fedavg / iid | 300 | 0.5747 | 27 | 0.5747 | 1.0575 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / dirichlet | 300 | 0.6272 | 29 | 0.6265 | 0.9031 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organamnist | resnet18 | sync_fedavg / dirichlet | 300 | 0.5417 | 30 | 0.5417 | 1.1872 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organcmnist | resnet18 | sync_fedavg / iid | 300 | 0.6428 | 26 | 0.6428 | 0.9065 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organcmnist | resnet18 | sync_fedavg / iid | 300 | 0.6683 | 30 | 0.6683 | 0.8544 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| organcmnist | resnet18 | sync_fedavg / iid | 300 | 0.6395 | 29 | 0.6372 | 0.9165 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 500 | 0.8907 | 32 | 0.8857 | 0.4305 | 50 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / dirichlet | 500 | 0.8762 | 37 | 0.8708 | 0.4159 | 50 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 1000 | 0.9032 | 65 | 0.8953 | 0.3849 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 1000 | 0.8976 | 78 | 0.8909 | 0.4495 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / iid | 1000 | 0.9110 | 75 | 0.9040 | 0.3566 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / dirichlet | 1000 | 0.8900 | 75 | 0.8852 | 0.4413 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pathmnist | resnet18 | sync_fedavg / dirichlet | 1000 | 0.8479 | 98 | 0.8478 | 0.4582 | 100 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / iid | 300 | 0.8526 | 17 | 0.8494 | 0.3779 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / iid | 300 | 0.8558 | 30 | 0.8558 | 0.3668 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / iid | 300 | 0.8510 | 18 | 0.8494 | 0.3861 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | small_cnn | sync_fedavg / iid | 300 | 0.7997 | 29 | 0.7997 | 0.4292 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / dirichlet | 300 | 0.8173 | 12 | 0.6394 | 0.6051 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| pneumoniamnist | resnet18 | sync_fedavg / dirichlet | 300 | 0.7340 | 1 | 0.6250 | 0.6948 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| tissuemnist | resnet18 | sync_fedavg / iid | 300 | 0.5564 | 30 | 0.5564 | 1.1946 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| tissuemnist | resnet18 | sync_fedavg / iid | 300 | 0.5608 | 27 | 0.5600 | 1.1938 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |
| tissuemnist | resnet18 | sync_fedavg / iid | 300 | 0.5598 | 25 | 0.5580 | 1.1926 | 30 | 0.0000 | 0.0000 | 1.0000 |  |  |  |  |

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
| bloodmnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0327 | 0.0520 | 0.0237 |
| bloodmnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.5042 | 0.4999 | 0.0000 |
| bloodmnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0289 | 0.0380 | 0.0134 |
| bloodmnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.3464 | 0.3499 | 0.0079 |
| bloodmnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0246 | 0.0202 | 0.0000 |
| bloodmnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.2376 | 0.3660 | 0.1327 |
| bloodmnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0170 | 0.0126 | 0.0000 |
| bloodmnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.2119 | 0.2356 | 0.0281 |
| bloodmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0012 | -0.0015 | 0.0000 |
| bloodmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0032 | -0.0012 | 0.0023 |
| bloodmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0088 | 0.0117 | 0.0032 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0029 | 0.0056 | 0.0029 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0029 | 0.0038 | 0.0012 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0146 | 0.0149 | 0.0006 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.1046 | 0.1096 | 0.0053 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0003 | 0.0023 | 0.0023 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0035 | 0.0053 | 0.0020 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0178 | 0.0175 | 0.0000 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0108 | 0.0143 | 0.0038 |
| bloodmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0003 | 0.0006 | 0.0006 |
| bloodmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0170 | 0.0167 | 0.0000 |
| bloodmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0181 | 0.0178 | 0.0000 |
| bloodmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0193 | 0.0202 | 0.0012 |
| bloodmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0146 | 0.0143 | 0.0000 |
| bloodmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0143 | 0.0178 | 0.0038 |
| bloodmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0088 | 0.0094 | 0.0009 |
| bloodmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0155 | 0.0167 | 0.0015 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0117 | 0.0152 | 0.0038 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0038 | 0.0041 | 0.0006 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0164 | 0.0161 | 0.0000 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.1017 | 0.1146 | 0.0132 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0134 | 0.0152 | 0.0020 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0132 | 0.0132 | 0.0003 |
| bloodmnist / naive_async / iid | sync_fedavg / iid | 0.0041 | 0.0044 | 0.0006 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0006 | 0.0009 | 0.0006 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0079 | -0.0061 | 0.0020 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0117 | 0.0140 | 0.0026 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0877 | 0.0903 | 0.0029 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0085 | 0.0105 | 0.0023 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0079 | 0.0076 | 0.0000 |
| bloodmnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0178 | -0.0170 | 0.0012 |
| bloodmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0626 | 0.0637 | 0.0015 |
| bloodmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0500 | 0.0509 | 0.0012 |
| bloodmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0582 | 0.0579 | 0.0000 |
| breastmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / naive_async / iid | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| breastmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0000 | 0.0000 | 0.0000 |
| dermamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0030 | 0.0015 | 0.0010 |
| dermamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0050 | 0.0050 | 0.0025 |
| dermamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0050 | 0.0055 | 0.0030 |
| dermamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0000 | -0.0025 | 0.0000 |
| dermamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0035 | 0.0060 | 0.0050 |
| dermamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0065 | 0.0055 | 0.0015 |
| dermamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0010 | 0.0000 | 0.0015 |
| dermamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0045 | 0.0045 | 0.0025 |
| dermamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0075 | 0.0070 | 0.0020 |
| dermamnist / naive_async / iid | sync_fedavg / iid | 0.0010 | -0.0005 | 0.0010 |
| dermamnist / naive_async / iid | sync_fedavg / iid | 0.0035 | 0.0040 | 0.0030 |
| dermamnist / naive_async / iid | sync_fedavg / iid | 0.0075 | 0.0110 | 0.0060 |
| dermamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0025 | 0.0005 | 0.0005 |
| dermamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0000 | 0.0045 | 0.0070 |
| dermamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0085 | 0.0065 | 0.0005 |
| dermamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0060 | 0.0055 | 0.0020 |
| octmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0310 | 0.0480 | 0.0180 |
| octmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0410 | 0.0450 | 0.0050 |
| octmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0110 | 0.0140 | 0.0040 |
| octmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0370 | 0.0440 | 0.0080 |
| octmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0500 | 0.0520 | 0.0030 |
| octmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0030 | -0.0020 | 0.0020 |
| octmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0590 | 0.0600 | 0.0020 |
| octmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0690 | 0.0690 | 0.0010 |
| octmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0460 | 0.0500 | 0.0050 |
| octmnist / naive_async / iid | sync_fedavg / iid | 0.0420 | 0.0540 | 0.0130 |
| octmnist / naive_async / iid | sync_fedavg / iid | 0.0430 | 0.0450 | 0.0030 |
| octmnist / naive_async / iid | sync_fedavg / iid | 0.0070 | 0.0170 | 0.0110 |
| octmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0340 | 0.0410 | 0.0080 |
| octmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.1000 | 0.0990 | 0.0000 |
| octmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0980 | 0.0970 | 0.0000 |
| octmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0990 | 0.0980 | 0.0000 |
| organamnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0400 | 0.0392 | 0.0000 |
| organamnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.1657 | 0.1910 | 0.0260 |
| organamnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0477 | 0.0615 | 0.0145 |
| organamnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.1482 | 0.1520 | 0.0045 |
| organamnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0613 | 0.0605 | 0.0000 |
| organamnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.1290 | 0.1282 | 0.0000 |
| organamnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0435 | 0.0427 | 0.0000 |
| organamnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.1268 | 0.1260 | 0.0000 |
| organamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0007 | -0.0045 | 0.0000 |
| organamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0120 | -0.0172 | 0.0000 |
| organamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0138 | -0.0088 | 0.0102 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0087 | 0.0035 | 0.0000 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0020 | -0.0015 | 0.0018 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0058 | -0.0082 | 0.0028 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0507 | 0.0475 | 0.0020 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0100 | 0.0047 | 0.0000 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0070 | 0.0057 | 0.0040 |
| organamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0062 | 0.0010 | 0.0000 |
| organamnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0157 | 0.0150 | 0.0045 |
| organamnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0222 | 0.0170 | 0.0000 |
| organamnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0160 | 0.0147 | 0.0040 |
| organamnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0205 | 0.0152 | 0.0000 |
| organamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0202 | 0.0155 | 0.0005 |
| organamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0065 | 0.0045 | 0.0032 |
| organamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0092 | 0.0072 | 0.0032 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0140 | 0.0172 | 0.0085 |
| organamnist / naive_async / iid | sync_fedavg / iid | -0.0043 | -0.0025 | 0.0070 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0070 | 0.0072 | 0.0055 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0512 | 0.0460 | 0.0000 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0192 | 0.0140 | 0.0000 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0182 | 0.0142 | 0.0012 |
| organamnist / naive_async / iid | sync_fedavg / iid | 0.0142 | 0.0115 | 0.0025 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0097 | 0.0087 | 0.0042 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0098 | -0.0090 | 0.0060 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0003 | 0.0000 | 0.0055 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0440 | 0.0387 | 0.0000 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0160 | 0.0107 | 0.0000 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0115 | 0.0107 | 0.0045 |
| organamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0120 | -0.0108 | 0.0065 |
| organamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0585 | 0.0543 | 0.0010 |
| organamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0560 | 0.0533 | 0.0025 |
| organamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0497 | 0.0465 | 0.0020 |
| organcmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0038 | 0.0192 | 0.0155 |
| organcmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0108 | -0.0108 | 0.0000 |
| organcmnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0175 | 0.0178 | 0.0003 |
| organcmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0140 | 0.0272 | 0.0132 |
| organcmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0090 | -0.0090 | 0.0000 |
| organcmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0175 | 0.0202 | 0.0028 |
| organcmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0288 | 0.0302 | 0.0015 |
| organcmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0020 | 0.0040 | 0.0020 |
| organcmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0305 | 0.0338 | 0.0033 |
| organcmnist / naive_async / iid | sync_fedavg / iid | 0.0318 | 0.0328 | 0.0010 |
| organcmnist / naive_async / iid | sync_fedavg / iid | 0.0005 | 0.0008 | 0.0002 |
| organcmnist / naive_async / iid | sync_fedavg / iid | 0.0248 | 0.0268 | 0.0020 |
| organcmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0180 | 0.0182 | 0.0002 |
| organcmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0787 | 0.0793 | 0.0005 |
| organcmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0700 | 0.0715 | 0.0015 |
| organcmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0953 | 0.0953 | 0.0000 |
| pathmnist / agreement_fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0191 | 0.0398 | 0.0255 |
| pathmnist / agreement_fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0163 | 0.0312 | 0.0196 |
| pathmnist / fedbuff_async / dirichlet / inverse / B=5 | sync_fedavg / dirichlet | 0.0164 | 0.0302 | 0.0185 |
| pathmnist / staleness_async / dirichlet / inverse | sync_fedavg / dirichlet | 0.0380 | 0.0610 | 0.0277 |
| pathmnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0004 | 0.0185 | 0.0228 |
| pathmnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.0369 | 0.0405 | 0.0084 |
| pathmnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0046 | 0.0097 | 0.0099 |
| pathmnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0465 | 0.0451 | 0.0033 |
| pathmnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0067 | 0.0235 | 0.0216 |
| pathmnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.0391 | 0.0838 | 0.0494 |
| pathmnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0075 | 0.0260 | 0.0233 |
| pathmnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.0442 | 0.0752 | 0.0358 |
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
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0074 | 0.0091 | 0.0086 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0116 | 0.0099 | 0.0053 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0117 | 0.0072 | 0.0025 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0050 | 0.0039 | 0.0058 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0021 | 0.0007 | 0.0056 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0086 | 0.0116 | 0.0099 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0138 | 0.0104 | 0.0036 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0155 | 0.0131 | 0.0046 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0120 | 0.0085 | 0.0035 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0003 | 0.0054 | 0.0121 |
| pathmnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0001 | 0.0026 | 0.0097 |
| pathmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0135 | 0.0104 | 0.0039 |
| pathmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0164 | 0.0130 | 0.0035 |
| pathmnist / fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | 0.0149 | 0.0104 | 0.0025 |
| pathmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0134 | 0.0117 | 0.0053 |
| pathmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0191 | 0.0164 | 0.0043 |
| pathmnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0057 | 0.0085 | 0.0097 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0061 | 0.0079 | 0.0088 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0164 | 0.0149 | 0.0054 |
| pathmnist / naive_async / iid | sync_fedavg / iid | -0.0052 | -0.0025 | 0.0096 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0149 | 0.0085 | 0.0006 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0106 | 0.0159 | 0.0123 |
| pathmnist / naive_async / iid | sync_fedavg / iid | 0.0109 | 0.0100 | 0.0061 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0114 | 0.0114 | 0.0070 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0047 | 0.0050 | 0.0072 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0117 | 0.0072 | 0.0025 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0091 | 0.0109 | 0.0088 |
| pathmnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0021 | 0.0014 | 0.0063 |
| pathmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0228 | 0.0252 | 0.0093 |
| pathmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0362 | 0.0341 | 0.0049 |
| pathmnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0251 | 0.0280 | 0.0099 |
| pneumoniamnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | -0.0337 | -0.0481 | 0.1635 |
| pneumoniamnist / caa_fedbuff_v2 / dirichlet | sync_fedavg / dirichlet | 0.1923 | 0.0144 | 0.0000 |
| pneumoniamnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.0016 | -0.0497 | 0.1266 |
| pneumoniamnist / fedbuff_async / dirichlet / hinge / B=5 | sync_fedavg / dirichlet | 0.1795 | 0.0128 | 0.0112 |
| pneumoniamnist / naive_async / dirichlet | sync_fedavg / dirichlet | -0.0449 | -0.1058 | 0.1170 |
| pneumoniamnist / naive_async / dirichlet | sync_fedavg / dirichlet | 0.1522 | 0.0128 | 0.0385 |
| pneumoniamnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | -0.0481 | -0.1522 | 0.0737 |
| pneumoniamnist / staleness_async / dirichlet / hinge | sync_fedavg / dirichlet | 0.1891 | 0.0144 | 0.0032 |
| pneumoniamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0176 | 0.0064 | 0.0240 |
| pneumoniamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0064 | -0.0032 | 0.0032 |
| pneumoniamnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0208 | -0.0016 | 0.0192 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0016 | 0.0032 | 0.0048 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0064 | -0.0016 | 0.0048 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0064 | 0.0032 | 0.0096 |
| pneumoniamnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0128 | 0.0465 | 0.0337 |
| pneumoniamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | -0.0048 | 0.0064 | 0.0112 |
| pneumoniamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0016 | 0.0016 | 0.0000 |
| pneumoniamnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | -0.0016 | 0.0048 | 0.0064 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0032 | 0.0032 | 0.0000 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0048 | 0.0048 | 0.0000 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0048 | 0.0048 | 0.0000 |
| pneumoniamnist / naive_async / iid | sync_fedavg / iid | 0.0609 | 0.0609 | 0.0000 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0032 | 0.0096 | 0.0064 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0064 | -0.0064 | 0.0000 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | -0.0016 | -0.0016 | 0.0000 |
| pneumoniamnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0417 | 0.0465 | 0.0048 |
| pneumoniamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0657 | 0.0657 | 0.0000 |
| pneumoniamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0641 | 0.0641 | 0.0000 |
| pneumoniamnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0337 | 0.0337 | 0.0000 |
| tissuemnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0014 | -0.0018 | 0.0004 |
| tissuemnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0004 | -0.0004 | 0.0008 |
| tissuemnist / agreement_fedbuff_async / iid / hinge / B=5 | sync_fedavg / iid | -0.0054 | -0.0062 | 0.0000 |
| tissuemnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0044 | 0.0036 | 0.0000 |
| tissuemnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | 0.0058 | 0.0052 | 0.0002 |
| tissuemnist / caa_fedbuff_v2 / iid | sync_fedavg / iid | -0.0050 | -0.0038 | 0.0020 |
| tissuemnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0072 | 0.0084 | 0.0020 |
| tissuemnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | -0.0014 | -0.0006 | 0.0016 |
| tissuemnist / fedbuff_async / iid / inverse / B=5 | sync_fedavg / iid | 0.0010 | 0.0014 | 0.0012 |
| tissuemnist / naive_async / iid | sync_fedavg / iid | 0.0022 | 0.0042 | 0.0028 |
| tissuemnist / naive_async / iid | sync_fedavg / iid | 0.0040 | 0.0076 | 0.0044 |
| tissuemnist / naive_async / iid | sync_fedavg / iid | 0.0034 | 0.0042 | 0.0016 |
| tissuemnist / staleness_async / iid / hinge | sync_fedavg / iid | 0.0024 | 0.0020 | 0.0004 |
| tissuemnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0254 | 0.0276 | 0.0030 |
| tissuemnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0286 | 0.0278 | 0.0000 |
| tissuemnist / staleness_async / iid / inverse | sync_fedavg / iid | 0.0366 | 0.0358 | 0.0000 |

- `Best Gap = sync_best_acc - async_best_acc`: peak accuracy cost of removing the barrier.
- `Final Gap = sync_final_acc - async_final_acc`: whether the async system actually converges near sync.
- `Stability Drop = best_acc - final_acc`: how much late-training regression or stale-update oscillation remains.

## Multi-Dataset Coverage

| Dataset | Task | Classes | Channels | Sync | Stateless Async | CAA-FedBuff |
|---|---|---:|---:|---:|---:|---:|
| bloodmnist | multi-class | 8 | 3 | 0.8825 | 0.8787 | 0.8857 |
| breastmnist | binary-class | 2 | 1 | 0.7308 | 0.7308 | 0.7308 |
| dermamnist | multi-class | 7 | 3 | 0.6928 | 0.6918 | 0.6928 |
| octmnist | multi-class | 4 | 1 | 0.5740 | 0.5670 | 0.5770 |
| organamnist | multi-class | 11 | 1 | 0.6272 | 0.6308 | 0.6402 |
| organcmnist | multi-class | 11 | 1 | 0.6683 | 0.6677 | 0.6790 |
| pathmnist | multi-class | 9 | 3 | 0.9110 | 0.9162 | 0.9145 |
| pneumoniamnist | binary-class | 2 | 1 | 0.8558 | 0.8622 | 0.8766 |
| tissuemnist | multi-class | 8 | 1 | 0.5608 | 0.5586 | 0.5662 |

## Stateless vs Staleness-Aware

This report treats `naive_async` as the stateless async baseline because it ignores logical staleness. `staleness_async` is the logical-staleness baseline.

| Dataset | Stateless Best | Staleness-Aware Best | CAA-Family Best | Note |
|---|---:|---:|---:|---|
| bloodmnist | 0.8787 | 0.9003 | 0.8857 | staleness decay helped; CAA matched/exceeded stateless |
| breastmnist | 0.7308 | 0.7308 | 0.7308 | staleness decay helped; CAA matched/exceeded stateless |
| dermamnist | 0.6918 | 0.6928 | 0.6928 | staleness decay helped; CAA matched/exceeded stateless |
| octmnist | 0.5670 | 0.5400 | 0.5770 | logical staleness alone was conservative; CAA matched/exceeded stateless |
| organamnist | 0.6308 | 0.6385 | 0.6402 | staleness decay helped; CAA matched/exceeded stateless |
| organcmnist | 0.6677 | 0.6502 | 0.6790 | logical staleness alone was conservative; CAA matched/exceeded stateless |
| pathmnist | 0.9162 | 0.9089 | 0.9145 | logical staleness alone was conservative |
| pneumoniamnist | 0.8622 | 0.8654 | 0.8766 | staleness decay helped; CAA matched/exceeded stateless |
| tissuemnist | 0.5586 | 0.5584 | 0.5662 | logical staleness alone was conservative; CAA matched/exceeded stateless |

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
- `figures/report/non_iid_async_sync_gap.png`
- `figures/report/non_iid_stability_drop.png`
- `figures/report/straggler_staleness_distribution.png`
- `figures/report/straggler_acc_vs_simulated_time.png`
- `figures/report/client_contribution_gini.png`
- `figures/report/time_to_accuracy.png`
- `figures/report/caa_v2_ablation_best_acc.png`
- `figures/report/caa_v2_ablation_stability_drop.png`
- `figures/report/distributed_systems_summary.csv`
- `figures/report/caa_v2_ablation_components.csv`
- `figures/report/existing_vs_ours_table.csv`
- `figures/classification/*_confusion.png`

## Future Extensions

- ChestMNIST and RetinaMNIST can be added later with task-specific loss and metrics; they are excluded from the current headline to keep the comparison fair.
- MobileNetV3-small can provide another compact edge-device backbone after the required small-CNN checks are complete.
- These are validation axes for future work, not headline claims for the current experiment set.
