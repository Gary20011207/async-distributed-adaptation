# LLM 

This repo runs a asynchronous federated learning framework optimized for Large Language Models (LLMs):

- **Dataset**: `MMLU` (Massive Multitask Language Understanding) benchmark
- **Model / Backbone**: `Qwen1.5-0.5B` (Autoregressive Large Language Model via HuggingFace Transformers)
- **Parameter-Efficient Tuning**: `PEFT LoRA` (Low-Rank Adaptation) matrix optimization
- **Clients**: 10 active edge clients (simulating heterogeneous hardware and network compute nodes)
- **Partition**: Standard IID or Dirichlet non-IID token/topic skew distribution
- **Methods**: `sync_fedavg`, `naive_async`, `staleness_async`, `fedbuff_async`, `agreement_fedbuff_async`, `caa_fedbuff_v2`
- **Supported Architectures**: `qwen` flagship track (leveraging specialized dynamic virtual-tensor structures under `--synthetic` for lightweight local sanity verification)


## Setup

Recommended isolated environment:

```bash
cd llm/pathMNIST
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
cd llm/pathMNIST
BASE_ENV=a2a_local ENV_NAME=fedpath-r139 bash scripts/bootstrap_env.sh
```

If your default Python is too new for a dependency wheel, use Python 3.11 or 3.12.
The supported Python range in `pyproject.toml` is `>=3.10,<3.14`.

## Smoke test

```bash
PYTHONPATH=src python src/fed_pathmnist/run.py --synthetic --rounds 1 --max-train-samples 200 --max-test-samples 100
fed-pathmnist --dataset mmlu --model qwen --method sync_fedavg --rounds 1 --max-train-samples 1000 --max-test-samples 300
```

## Benchmark & Automated Workloads

This project relies on a dual-dispatcher automated testing pipeline. Instead of running manual single-seed commands, all benchmarks, network stress tests, and ablation studies are executed via two robust 24-hour background runners.

To launch the complete evaluation suite, execute the following dispatchers:

```bash
# Track A: Launch Distributed Systems & Extreme Network Stress Test
nohup bash scripts/run_distributed_systems_24h.sh > logs/distsys24_nohup.log 2>&1 &

# Track B: Launch IID Baselines & Automated Report Compiler
nohup bash scripts/run_report_completion_24h.sh > logs/report24_nohup.log 2>&1 &
```

## Analytics
Once the background runners or manual optimization tracks conclude, execute the parsing sequence to recalculate multi-seed means $\pm$ standard deviations and instantly refresh the academic artifacts:
```bash
python scripts/plot_report_summary.py --result-dir results --outdir figures/report --model qwen
python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report --model qwen
python scripts/plot_distributed_systems_summary.py --result-dir results --outdir figures/report --model qwen
python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md
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
figures/report/distributed_systems_summary.csv
figures/report/caa_v2_ablation_components.csv
```

The core presentation arguments utilize three primary metrics to assess non-synchronous optimization dynamics:
```text
Async-Sync Best Gap = sync_best_acc - async_best_acc
Async-Sync Final Gap = sync_final_acc - async_final_acc
Stability Drop = best_acc - final_acc
```

Generated artifact policy:

```text
results/, figures/, logs/, checkpoints/, data/, __pycache__/, and *.egg-info/ are ignored.
The pushable summary is ../REPORT_NOTES.md; raw logs, downloaded data, and checkpoints should stay local.
```