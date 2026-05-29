# Experiment Completeness Audit

Last checked: 2026-05-24 05:16 Asia/Taipei.

## Current Conclusion

The **full ResNet18 IID fair-budget matrix is complete**.

- Primary datasets: `pathmnist`, `pneumoniamnist`, `bloodmnist`, `organamnist`
- Extended datasets: `dermamnist`, `octmnist`, `breastmnist`, `tissuemnist`, `organcmnist`
- Model: `resnet18`
- Partition: `iid`
- Seeds: `42, 43, 44`
- Required methods: `sync_fedavg`, `naive_async`, `staleness_async`,
  `fedbuff_async`, `agreement_fedbuff_async`, `caa_fedbuff_v2`
- Fair budget:
  - `pathmnist`: Sync `100 rounds`, Async `1000 events`
  - Other datasets: Sync `30 rounds`, Async `300 events`

These rows support real `mean ± std` claims for every official dataset/method
cell.

Current summary count: `302` summary JSON files.

## Full Baseline Matrix Completion

Completed on 2026-05-24 04:36 Asia/Taipei.

The stricter full headline matrix has been completed:

```text
datasets = pathmnist, pneumoniamnist, bloodmnist, organamnist,
           dermamnist, octmnist, breastmnist, tissuemnist, organcmnist
methods  = sync_fedavg, naive_async, staleness_async,
           fedbuff_async, agreement_fedbuff_async, caa_fedbuff_v2
seeds    = 42, 43, 44
model    = resnet18
partition = iid
fair budget = async events = sync rounds * clients
```

Completion runner:

```text
r13946001/pathMNIST/scripts/run_full_fair_matrix_completion.sh
```

Transient service:

```text
fedpath-r139-full-fair-matrix.service
```

Logs:

```text
r13946001/pathMNIST/logs/full_fair_matrix_completion_*.log
r13946001/pathMNIST/logs/full_fair_matrix_completion_heartbeat_*.log
```

Initial audit before launch found `35` missing runs, mostly `fedbuff_async`
and extended-dataset `agreement_fedbuff_async`. The runner skipped existing
summaries and executed only missing fair-budget runs.

Final heartbeat:

```text
progress=162/162
active_training=
summary_count=302
```

The transient service has exited and no `src/fed_pathmnist/run.py` training
process is active.

`figures/report/mean_std_summary.csv` now contains:

```text
rows = 54
datasets = 9
methods = 6
min_seed_count = 3
max_seed_count = 3
seed_count < 3 = none
```

## Missing-Seed Completion

A targeted runner was added:

```text
r13946001/pathMNIST/scripts/run_missing_seed_completion.sh
```

It filled the extended-dataset fair-budget matrix for:

```text
datasets = dermamnist, octmnist, breastmnist, tissuemnist, organcmnist
methods  = sync_fedavg, naive_async, staleness_async, caa_fedbuff_v2
seeds    = 42, 43, 44
budget   = 300 client updates
model    = resnet18
partition = iid
```

The runner skipped any summary JSON that already existed, so it did not rerun
completed experiments.

The completion runner finished successfully:

```text
Completed missing seed completion failures=0
```

The transient systemd service has exited and no `src/fed_pathmnist/run.py`
training process is active.

Logs:

```text
r13946001/pathMNIST/logs/missing_seed_completion_20260523_195005.log
r13946001/pathMNIST/logs/missing_seed_completion_heartbeat_20260523_195005.log
```

## Interpretation Guidance

Use this wording in the report:

> All reported ResNet18 IID datasets have matched update budget and three-seed
> mean ± std for Sync FedAvg, Naive Async, Staleness Async, FedBuff-lite,
> CAA-FedBuff, and CAA-v2.

## Correctness Checks

The current results are internally consistent with the project story:

- `async events = sync rounds * clients` for official fair-budget comparisons.
- Primary and extended ResNet18 IID rows have seed count `3` for all six
  official methods, so standard deviation is meaningful.
- The conclusion should remain conservative: CAA-v2 approaches Sync and improves
  some settings, but it does not dominate every baseline on every dataset.
- Non-IID, straggler, ablation, and backbone results are representative analysis
  layers; they are not claimed as a full factorial grid over every dataset.
