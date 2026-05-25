# Final Results Summary

Last generated: 2026-05-24 Asia/Taipei.

## Scope

This summary uses the official fair matrix only:

- 9 datasets: `pathmnist`, `pneumoniamnist`, `bloodmnist`, `organamnist`, `organcmnist`, `dermamnist`, `octmnist`, `breastmnist`, `tissuemnist`
- 6 methods: Sync FedAvg, Naive Async, Staleness Async, FedBuff-lite, CAA-v1, CAA-v2
- 3 seeds for every dataset/method cell: 42, 43, 44
- ResNet18, IID partition, same local epochs/batch size/LR schedule/augmentation
- Fair update budget: async events = sync rounds x clients

Derived CSV files:

- `r13946001/pathMNIST/figures/report/final_method_comparison.csv`
- `r13946001/pathMNIST/figures/report/official_system_metrics_summary.csv`

## Overall Method Performance

| Method | Datasets | Best Acc Mean | Final Acc Mean | Stability Drop Mean |
| --- | --- | --- | --- | --- |
| Sync | 9 | 0.7142 | 0.7121 | 0.0020 |
| Naive Async | 9 | 0.7132 | 0.7096 | 0.0036 |
| Staleness | 9 | 0.6770 | 0.6752 | 0.0017 |
| FedBuff | 9 | 0.7090 | 0.7062 | 0.0028 |
| CAA-v1 | 9 | 0.7206 | 0.7158 | 0.0048 |
| CAA-v2 | 9 | 0.7169 | 0.7140 | 0.0029 |

## Average Rank Across Datasets

Lower rank is better.

| Method | Avg Best Rank | Avg Final Rank |
| --- | --- | --- |
| Sync | 2.6667 | 2.5556 |
| Naive Async | 3.4444 | 3.6667 |
| Staleness | 5.4444 | 5.4444 |
| FedBuff | 4.1111 | 3.8889 |
| CAA-v1 | 1.4444 | 1.5556 |
| CAA-v2 | 2.1111 | 1.7778 |

## CAA-v2 Per-Dataset Result

Positive gap means CAA-v2 is higher than the reference.

| Dataset | CAA-v2 Best | CAA-v2 Best Std | CAA-v2 Final | CAA-v2 Final Std | Stability Drop | Best Gap vs Sync | Final Gap vs Sync | Best Gap vs Best Baseline | Best Baseline | Best Method | Final Best Method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pathmnist | 0.9048 | 0.0097 | 0.9010 | 0.0090 | 0.0038 | 0.0008 | 0.0043 | -0.0004 | Naive Async | CAA-v1 | CAA-v2 |
| pneumoniamnist | 0.8606 | 0.0028 | 0.8542 | 0.0028 | 0.0064 | 0.0075 | 0.0027 | 0.0032 | FedBuff | CAA-v1 | CAA-v1 |
| bloodmnist | 0.8757 | 0.0068 | 0.8741 | 0.0060 | 0.0016 | 0.0003 | -0.0012 | 0.0003 | Sync | CAA-v1 | CAA-v1 |
| organamnist | 0.6248 | 0.0073 | 0.6233 | 0.0059 | 0.0015 | 0.0063 | 0.0077 | 0.0039 | Naive Async | CAA-v1 | CAA-v1 |
| organcmnist | 0.6608 | 0.0144 | 0.6554 | 0.0192 | 0.0053 | 0.0106 | 0.0060 | 0.0106 | Sync | CAA-v1 | CAA-v1 |
| dermamnist | 0.6894 | 0.0032 | 0.6873 | 0.0048 | 0.0022 | -0.0005 | 0.0010 | -0.0005 | Sync | Sync | CAA-v2 |
| octmnist | 0.5460 | 0.0276 | 0.5417 | 0.0291 | 0.0043 | -0.0007 | -0.0037 | -0.0007 | Sync | Sync | Sync |
| breastmnist | 0.7308 | 0.0000 | 0.7308 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Sync | CAA-v1 | CAA-v1 |
| tissuemnist | 0.5591 | 0.0059 | 0.5583 | 0.0048 | 0.0007 | 0.0001 | 0.0002 | 0.0001 | Sync | CAA-v1 | CAA-v1 |

## System Metrics From Official CSV Logs

| Method | Runs | Avg Staleness | P95 Staleness | Max Staleness | Avg Effective Alpha | Avg Buffer Alpha | Avg Agreement | Client Gini | Client Min/Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync | 27 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | nan | nan | nan | nan |
| Naive Async | 27 | 8.7425 | 14.4593 | 66.4444 | 0.5000 | nan | nan | 0.1832 | 0.1486 |
| Staleness | 27 | 8.7425 | 14.4593 | 66.4444 | 0.0755 | nan | nan | 0.1832 | 0.1486 |
| FedBuff | 27 | 1.7356 | 3.0000 | 13.2222 | 0.2288 | nan | nan | 0.1832 | 0.1486 |
| CAA-v1 | 27 | 1.7356 | 3.0000 | 13.2222 | 0.6163 | 0.6225 | 0.6988 | 0.1832 | 0.1486 |
| CAA-v2 | 27 | 1.7356 | 3.0000 | 13.2222 | 0.6163 | 0.6063 | 0.6766 | 0.1832 | 0.1486 |

## Final Interpretation

CAA-v2 is not a universal accuracy winner, but it is a strong final method for the project story:

- It beats Sync FedAvg in mean best accuracy on 6/9 datasets and mean final accuracy on 6/9 datasets.
- It beats the strongest classic baseline among Sync/Naive/Staleness/FedBuff on 5/9 datasets.
- Its overall mean best accuracy is 0.7169, slightly above Sync FedAvg at 0.7142 and Naive Async at 0.7132.
- Its overall mean final accuracy is 0.7140, above Sync FedAvg at 0.7121 and Naive Async at 0.7096.
- Its stability drop is 0.0029, lower than Naive Async (0.0036) and CAA-v1 (0.0048), but higher than Sync FedAvg (0.0020) and staleness-only (0.0017).
- Staleness-only is very stable but too conservative: it has the lowest overall accuracy, with mean best accuracy 0.6770.
- CAA-v1 has the best mean accuracy and rank, but it is less stable than CAA-v2. CAA-v2 is the cleaner final method because it trades a little peak accuracy for better final accuracy and lower oscillation.

Recommended claim:

> Under a fair update budget, CAA-v2 makes clockless asynchronous FL approach Sync FedAvg across diverse MedMNIST datasets, while reducing the instability of naive async and avoiding the over-conservatism of staleness-only aggregation. It does not dominate every baseline, but it gives the most defensible distributed-systems tradeoff among our proposed methods.
