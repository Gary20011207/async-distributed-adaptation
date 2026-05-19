# R13946001 Safe Experiment Copy

This folder is an isolated copy of the original `pathMNIST/` project.

The goal is to let us try new final-project features without changing the
team's original baseline code at the repository root.

Current additions in this copy:

- CSV logging under `results/`
- summary JSON under `results/`
- optional best checkpoint saving under `checkpoints/`
- general staleness decay rules
- simulated delay modes
- FedBuff-lite buffered async aggregation
- plotting utility
- Dirichlet non-IID partitioning via `--partition dirichlet`
- fallback Flower client base class for local smoke tests before `flwr` is installed
- CAA-FedBuff (`agreement_fedbuff_async`) with agreement-aware buffered aggregation
- CAA-FedBuff v2 (`caa_fedbuff_v2`) with server trajectory agreement and client fairness credit
- multi-dataset 2D MedMNIST support via `--dataset`
- model backbone selection via `--model resnet18|small_cnn|mobilenet_v3_small`
- slide-friendly report plots under `figures/report/`
- multi-seed mean/std reporting and classification diagnostics
- 24-hour report-completion runner for extra datasets, small-CNN checks, and CAA-v2 ablations

Current completed dataset coverage:

```text
pathmnist, pneumoniamnist, bloodmnist, dermamnist, organamnist, octmnist,
breastmnist, tissuemnist, organcmnist
```

For the main coverage datasets, Sync FedAvg, stateless async
(`naive_async`), staleness-aware async, CAA-FedBuff, and CAA-v2 have completed runs.
The most important report file is:

```text
r13946001/REPORT_NOTES.md
```

The cleanest figure set for slides is:

```text
r13946001/pathMNIST/figures/report/
```

Run from this directory:

```bash
cd r13946001/pathMNIST
PYTHONPATH=src python src/fed_pathmnist/run.py --synthetic --method fedbuff_async --events 4 --clients 2 --buffer-size 2 --device cpu --save-best
```

Non-IID smoke test:

```bash
cd r13946001/pathMNIST
PYTHONPATH=src python src/fed_pathmnist/run.py --synthetic --method sync_fedavg --partition dirichlet --dirichlet-alpha 0.5 --rounds 1 --clients 2 --device cpu
```

Plot logs:

```bash
PYTHONPATH=src python -m fed_pathmnist.plot_results --csv results/*.csv --outdir figures
python scripts/plot_report_summary.py --result-dir results --outdir figures/report
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report
python scripts/plot_classification_results.py --result-dir results --checkpoint-dir checkpoints --outdir figures/classification
python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md
```

CAA-FedBuff v2 smoke test:

```bash
cd r13946001/pathMNIST
PYTHONPATH=src python src/fed_pathmnist/run.py --synthetic --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --model small_cnn --device cpu
```

Bootstrap the dedicated long-run environment:

```bash
cd r13946001/pathMNIST
BASE_ENV=a2a_local ENV_NAME=fedpath-r139 bash scripts/bootstrap_env.sh
```

Minimal setup without conda:

```bash
cd r13946001/pathMNIST
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If CUDA PyTorch wheels are not resolved automatically, install the matching
`torch` and `torchvision` wheels first, then rerun `python -m pip install -e .`.

Start the 24-hour report-completion runner:

```bash
cd r13946001/pathMNIST
nohup bash scripts/run_report_completion_24h.sh > logs/report24_nohup.log 2>&1 &
```

The runner writes:

```text
logs/report24_heartbeat_YYYYMMDD_HHMMSS.log
logs/report24_actions_YYYYMMDD_HHMMSS.log
```

Push hygiene:

```text
results/, figures/, logs/, checkpoints/, data/, __pycache__/, and *.egg-info/ are ignored.
Commit the code, README files, SURVEY_NOTES.md, and REPORT_NOTES.md.
Do not force-add raw logs, downloaded data, or model checkpoints.
```
