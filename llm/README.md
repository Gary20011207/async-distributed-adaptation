# LLM Safe Experiment Copy

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
- model backbone selection via `--model mmlu`
- slide-friendly report plots under `figures/report/`
- multi-seed mean/std reporting and classification diagnostics
- 24-hour report-completion runner for extra datasets, small-CNN checks, and CAA-v2 ablations
- distributed-systems 24-hour runner for non-IID hospital scenarios, straggler stress tests, CAA-v2 ablations, and system metrics beyond accuracy

Current completed dataset coverage: mmlu



For the main coverage datasets, Sync FedAvg, stateless async
(`naive_async`), staleness-aware async, CAA-FedBuff, and CAA-v2 have completed runs.
The most important report file is:

```text
llm/REPORT_NOTES.md
```

The cleanest figure set for slides is:

```text
llm/pathMNIST/figures/report/
```

Run from this directory:

```bash
cd llm/pathMNIST

# 1. Asynchronous Bounded Buffer Smoke-Test (2 clients, synthetic tokens, CPU bound)
PYTHONPATH=src python src/fed_qwenllm/run.py --synthetic --method fedbuff_async --events 4 --clients 2 --buffer-size 2 --device cpu --save-best

# 2. Non-IID Dirichlet Partition Smoke-Test (Statistical topic skew, Dirichlet alpha = 0.5)
PYTHONPATH=src python src/fed_qwenllm/run.py --synthetic --method sync_fedavg --partition dirichlet --dirichlet-alpha 0.5 --rounds 1 --clients 2 --device cpu

# 3. CAA-FedBuff v2 Alignment Smoke-Test (Lightweight CNN verification backbone)
PYTHONPATH=src python src/fed_qwenllm/run.py --synthetic --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --model small_cnn --device cpu
```


Plot logs:

```bash
cd llm/pathMNIST

# Render active trajectory consensus and cross-client directional agreement maps
PYTHONPATH=src python -m fed_qwenllm.plot_results --csv results/*.csv --outdir figures

# Refresh presentation artifacts and compile the centralized results table
python scripts/plot_report_summary.py --result-dir results --outdir figures/report
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report
python scripts/plot_distributed_systems_summary.py --result-dir results --outdir figures/report
python scripts/plot_classification_results.py --result-dir results --checkpoint-dir checkpoints --outdir figures/classification
python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md
```

Bootstrap the dedicated long-run environment:

```bash
cd llm/pathMNIST
BASE_ENV=a2a_local ENV_NAME=fedpath-r139 bash scripts/bootstrap_env.sh
```

Minimal setup without conda:

```bash
cd llm/pathMNIST
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If CUDA PyTorch wheels are not resolved automatically, install the matching
`torch` and `torchvision` wheels first, then rerun `python -m pip install -e .`.

Start the 24-hour report-completion runner:

```bash
cd llm/pathMNIST
nohup bash scripts/run_report_completion_24h.sh > logs/report24_nohup.log 2>&1 &
```

The runner writes:

```text
logs/report24_heartbeat_YYYYMMDD_HHMMSS.log
logs/report24_actions_YYYYMMDD_HHMMSS.log
```

Start the distributed-systems-focused 24-hour runner:

```bash
cd llm/pathMNIST
nohup bash scripts/run_distributed_systems_24h.sh > logs/distsys24_nohup.log 2>&1 &
```

The runner writes heartbeats and keeps running idle QA
heartbeats even if its experiment queue finishes early:

```text
logs/distsys24_heartbeat_YYYYMMDD_HHMMSS.log
logs/distsys24_actions_YYYYMMDD_HHMMSS.log
```

Push hygiene:

```text
results/, figures/, logs/, checkpoints/, data/, __pycache__/, and *.egg-info/ are ignored.
Commit the code, README files, SURVEY_NOTES.md, and REPORT_NOTES.md.
Do not force-add raw logs, downloaded data, or model checkpoints.
```
