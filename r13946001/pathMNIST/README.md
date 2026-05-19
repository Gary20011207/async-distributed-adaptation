# PathMNIST Federated Learning Starter

This repo runs a first working version of the final project:

- Dataset: PathMNIST by default, with 2D MedMNIST `--dataset` support
- Model: ResNet18
- Clients: 10
- Partition: IID or Dirichlet non-IID
- Methods: `sync_fedavg`, `naive_async`, `staleness_async`, `fedbuff_async`, `agreement_fedbuff_async`, `caa_fedbuff_v2`
- Backbones: `resnet18`, `small_cnn`, `mobilenet_v3_small`
- Client API: Flower `NumPyClient`

This is the `r13946001` safe experiment copy. The original team baseline at
the repository root is intentionally not modified by these experiments.

## Setup

Recommended isolated environment:

```bash
cd r13946001/pathMNIST
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If PyTorch CUDA wheels are not resolved correctly by your platform, install
the matching `torch`/`torchvision` pair from the official PyTorch selector
first, then run `python -m pip install -e .`.

On our experiment machine we used a conda environment named `fedpath-r139`.
If a local base environment with PyTorch already exists, the helper can clone
it and install the project:

```bash
cd r13946001/pathMNIST
BASE_ENV=a2a_local ENV_NAME=fedpath-r139 bash scripts/bootstrap_env.sh
```

If your default Python is too new for a dependency wheel, use Python 3.11 or 3.12.
The supported Python range in `pyproject.toml` is `>=3.10,<3.14`.

## Smoke test

Run a tiny synthetic test first:

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py \
  --synthetic --rounds 1 --max-train-samples 200 --max-test-samples 100 \
  --result-dir /tmp/fedpath_smoke_results --checkpoint-dir /tmp/fedpath_smoke_checkpoints
```

Run PathMNIST with a small subset:

```bash
fed-pathmnist --method sync_fedavg --rounds 1 --max-train-samples 1000 --max-test-samples 300
```

Run another 2D MedMNIST dataset:

```bash
fed-pathmnist --dataset pneumoniamnist --method sync_fedavg --rounds 30 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --device cuda --save-best
```

Run with train-time augmentation and a cosine LR scheduler:

```bash
fed-pathmnist --method sync_fedavg --rounds 100 --augment --lr-scheduler cosine --device cuda
```

Try async variants:

```bash
fed-pathmnist --method naive_async --events 20 --max-train-samples 1000 --max-test-samples 300
fed-pathmnist --method staleness_async --events 20 --max-train-samples 1000 --max-test-samples 300
```

Run the new buffered async variant:

```bash
fed-pathmnist --method fedbuff_async --events 1000 --clients 10 --buffer-size 5 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 20 --device cuda --alpha 0.5 --staleness-decay inverse --delay-mode heterogeneous --save-best
```

Run the course-project CAA-FedBuff variant:

```bash
fed-pathmnist --method agreement_fedbuff_async --events 1000 --clients 10 --buffer-size 5 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 20 --device cuda --alpha 0.55 --staleness-decay hinge --staleness-hinge-a 0.05 --delay-mode heterogeneous --save-best
```

Run the CAA-FedBuff v2 variant:

```bash
fed-pathmnist --method caa_fedbuff_v2 --events 1000 --clients 10 --buffer-size 5 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --eval-every 20 --device cuda --alpha 0.62 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --agreement-epsilon 0.15 --agreement-power 0.5 --delta-clip-multiplier 1.8 --adaptive-alpha-min 0.20 --adaptive-alpha-max 0.70 --adaptive-staleness-scale 10 --server-delta-momentum 0.8 --history-agreement-blend 0.25 --client-fairness-power 0.5 --delay-mode heterogeneous --save-best
```

CAA-v2 adds two clockless signals on top of CAA-FedBuff:

```text
server accepted-delta EMA: recent global update direction
client fairness credit: downweights clients that have already dominated recent accepted updates
```

CAA-FedBuff logs additional agreement metrics:

```text
agreement, mean_agreement, buffer_alpha, delta_norm, dropped_update
```

All methods now write structured CSV logs and summary JSON files:

```text
results/<method>_pathmnist_YYYYMMDD_HHMMSS.csv
results/<method>_pathmnist_YYYYMMDD_HHMMSS_summary.json
```

Use `--save-best` to save the best evaluated model:

```text
checkpoints/<method>_pathmnist_YYYYMMDD_HHMMSS_best.pt
```

Plot one or more CSV logs:

```bash
python -m fed_pathmnist.plot_results --csv results/*.csv --outdir figures
```

This also creates:

```text
figures/agreement_vs_event.png
```

Create slide-friendly dataset-level figures and refresh the report notes:

```bash
python scripts/plot_report_summary.py --result-dir results --outdir figures/report
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report
python scripts/plot_classification_results.py --result-dir results --checkpoint-dir checkpoints --outdir figures/classification
python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md
```

For non-default backbone summaries, pass the same model filter, for example:

```bash
python scripts/plot_report_summary.py --result-dir results --outdir figures/report --model small_cnn
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report --model small_cnn
```

This creates:

```text
figures/report/best_accuracy_by_dataset.png
figures/report/final_accuracy_by_dataset.png
figures/report/async_sync_best_gap_by_dataset.png
figures/report/stability_drop_by_dataset.png
figures/report/caa_gap_and_stability_by_dataset.png
figures/report/dataset_method_heatmap.png
figures/report/best_method_summary.csv
figures/report/seeded_summary.csv
figures/report/mean_std_summary.csv
figures/report/fairness_protocol.csv
figures/report/existing_vs_ours_table.csv
```

Use these report metrics for the distributed-systems story:

```text
Async-Sync Best Gap = sync_best_acc - async_best_acc
Async-Sync Final Gap = sync_final_acc - async_final_acc
Stability Drop = best_acc - final_acc
```

Run a simple non-IID Dirichlet split:

```bash
fed-pathmnist --method sync_fedavg --partition dirichlet --dirichlet-alpha 0.5 --rounds 100 --clients 10 --batch-size 128 --lr 0.01 --lr-scheduler cosine --min-lr 0.0001 --local-epochs 1 --augment --device cuda --save-best
```

Use full data by omitting `--max-train-samples` and `--max-test-samples`.

Start the 24-hour report-completion runner:

```bash
nohup bash scripts/run_report_completion_24h.sh > logs/report24_nohup.log 2>&1 &
```

This runner refreshes the current report pack, adds BreastMNIST/TissueMNIST
single-seed fair coverage, runs `small_cnn` backbone checks, and performs
CAA-v2 ablations for server trajectory agreement and client fairness credit.

Generated artifact policy:

```text
results/, figures/, logs/, checkpoints/, data/, __pycache__/, and *.egg-info/ are ignored.
The pushable summary is ../REPORT_NOTES.md; raw logs, downloaded data, and checkpoints should stay local.
```
