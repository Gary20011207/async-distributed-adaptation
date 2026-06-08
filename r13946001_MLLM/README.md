# R13946001_MLLM: Clockless Federated Adaptation for LLM/MLLM

This folder is an isolated LLM/MLLM-only experiment copy. It does not modify the root project or the earlier `r13946001/` MedMNIST work.

The project question is:

```text
In no-global-clock asynchronous federated adaptation, can a server aggregate stale LLM/MLLM adapter updates while approaching Sync FedAvg under a fair update budget?
```

## What This Copy Contains

- Package: `fed_mllm`
- CLI: `fed-mllm` or `PYTHONPATH=src python src/fed_mllm/run.py`
- Text closed-ended QA adapters: `mmlu`, `medmcqa`, `medqa_usmle`, `pubmedqa`
- Medical closed-ended VQA adapters: `vqa_rad_closed`, `path_vqa_closed`
- Methods: `sync_fedavg`, `naive_async`, `staleness_async`, `fedbuff_async`, `agreement_fedbuff_async`, `caa_fedbuff_v2`
- Models:
  - `tiny_text`: dependency-light text smoke model
  - `tiny_vqa`: dependency-light image+text smoke model
  - `qwen_vl_proxy`: compact closed-ended VQA proxy used by the earlier VQA matrix
  - `qwen_text_0_5b_lora`: Qwen1.5-0.5B sequence classification with LoRA
  - `qwen2_5_vl_3b_qlora`: real Qwen2.5-VL-3B-Instruct 4-bit QLoRA generative closed-answer path
- Sequential client trainer: clients do not keep 10 copies of Qwen in memory; each client creates, trains, extracts adapter parameters, then releases the model.
- CSV logging and summary JSON under `results/`
- 24-hour heartbeat runner under `scripts/run_mllm_24h.sh`

## Setup

Recommended on the experiment machine:

```bash
cd r13946001_MLLM
BASE_ENV=fedpath-r139 ENV_NAME=fed-mllm-r139 bash scripts/bootstrap_env.sh
conda activate fed-mllm-r139
```

Minimal setup:

```bash
cd r13946001_MLLM
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If PyTorch CUDA wheels are not resolved automatically, install the matching `torch` and `torchvision` wheels first.

## Smoke Tests

Dependency-light text MCQA CAA-v2 smoke test:

```bash
cd r13946001_MLLM
PYTHONPATH=src python src/fed_mllm/run.py \
  --synthetic --task text_mcqa --dataset mmlu --model tiny_text \
  --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --device cpu \
  --result-dir /tmp/fed_mllm_smoke_results --checkpoint-dir /tmp/fed_mllm_smoke_checkpoints
```

Dependency-light medical VQA smoke test:

```bash
PYTHONPATH=src python src/fed_mllm/run.py \
  --synthetic --task medical_vqa --dataset vqa_rad_closed --model tiny_vqa \
  --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --device cpu \
  --result-dir /tmp/fed_mllm_smoke_results --checkpoint-dir /tmp/fed_mllm_smoke_checkpoints
```

Tiny CUDA check:

```bash
PYTHONPATH=src python src/fed_mllm/run.py \
  --synthetic --task medical_vqa --dataset vqa_rad_closed --model tiny_vqa \
  --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 \
  --max-train-samples 8 --max-test-samples 8 --device cuda \
  --result-dir /tmp/fed_mllm_smoke_results --checkpoint-dir /tmp/fed_mllm_smoke_checkpoints
```

Real Qwen2.5-VL QLoRA smoke test:

```bash
PYTHONPATH=src python src/fed_mllm/run.py \
  --synthetic --task medical_vqa --dataset vqa_rad_closed --model qwen2_5_vl_3b_qlora \
  --method sync_fedavg --rounds 1 --clients 1 --batch-size 1 \
  --max-train-samples 1 --max-test-samples 1 --max-length 128 --device cuda \
  --result-dir /tmp/fed_mllm_qwenvl_smoke_results \
  --checkpoint-dir /tmp/fed_mllm_qwenvl_smoke_checkpoints
```

This path loads the official 3B VLM with 4-bit quantization and trains only LoRA adapters.
Use very small samples first because sequential FL still reloads the VLM for each client job.

## Example Real Runs

Text MCQA baseline:

```bash
fed-mllm --task text_mcqa --dataset medmcqa --model qwen_text_0_5b_lora \
  --method sync_fedavg --rounds 10 --clients 10 --batch-size 4 \
  --max-train-samples 3000 --max-test-samples 1000 --device cuda --save-best
```

Text MCQA CAA-v2:

```bash
fed-mllm --task text_mcqa --dataset medmcqa --model qwen_text_0_5b_lora \
  --method caa_fedbuff_v2 --events 100 --clients 10 --batch-size 4 \
  --max-train-samples 3000 --max-test-samples 1000 --device cuda --delay-mode heterogeneous \
  --alpha 0.62 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 \
  --buffer-size 5 --agreement-epsilon 0.15 --agreement-power 0.5 \
  --delta-clip-multiplier 1.8 --adaptive-alpha-min 0.20 --adaptive-alpha-max 0.70 \
  --adaptive-staleness-scale 10 --server-delta-momentum 0.8 \
  --history-agreement-blend 0.25 --client-fairness-power 0.5 --save-best
```

Medical VQA compact proxy:

```bash
fed-mllm --task medical_vqa --dataset vqa_rad_closed --model qwen_vl_proxy \
  --method caa_fedbuff_v2 --events 50 --clients 10 --batch-size 1 \
  --gradient-accumulation-steps 4 --max-train-samples 1000 --max-test-samples 300 \
  --device cuda --delay-mode heterogeneous --save-best
```

## Long Runners

Proxy/text 24-hour runner:

```bash
cd r13946001_MLLM
nohup bash scripts/run_mllm_24h.sh > logs/mllm24_nohup.log 2>&1 &
```

Real Qwen2.5-VL + additional closed QA runner:

```bash
cd r13946001_MLLM
nohup bash scripts/run_qwenvl_closed_qa_12h.sh > logs/qwenvl12_nohup.log 2>&1 &
```

Focused report-polish runner after the training queue is complete:

```bash
cd r13946001_MLLM
nohup bash scripts/run_mllm_report_polish_3h.sh > logs/report_polish3h_nohup.log 2>&1 &
```

Balanced 24-hour MLLM completion runner:

```bash
cd r13946001_MLLM
systemd-run --user --unit fed-mllm-complete24-$(date +%Y%m%d-%H%M) bash scripts/run_mllm_completion_24h.sh
```

This 3-hour runner does not launch another full training matrix. It refreshes tables, plots, audit notes, the Chinese HTML report, and small Qwen2.5-VL closed-answer diagnostics.

The long runners write:

```text
logs/mllm24_heartbeat_YYYYMMDD_HHMMSS.log
logs/mllm24_actions_YYYYMMDD_HHMMSS.log
logs/mllm24_failures_YYYYMMDD_HHMMSS.log
logs/qwenvl12_heartbeat_YYYYMMDD_HHMMSS.log
logs/qwenvl12_actions_YYYYMMDD_HHMMSS.log
```

The report-polish runner writes:

```text
logs/report_polish3h_heartbeat_YYYYMMDD_HHMMSS.log
logs/report_polish3h_actions_YYYYMMDD_HHMMSS.log
logs/mllm_complete24_heartbeat_YYYYMMDD_HHMMSS.log
logs/mllm_complete24_actions_YYYYMMDD_HHMMSS.log
logs/mllm_complete24_failures_YYYYMMDD_HHMMSS.log
```

Heartbeat behavior:

```text
24-hour runner: 144 cycles, one every 10 minutes
12-hour runner: 72 cycles, one every 10 minutes
3-hour report runner: 18 cycles, one every 10 minutes
no overlapping CUDA jobs
failed runs are logged and skipped
```

## Report Pack

```bash
PYTHONPATH=src python -m fed_mllm.plot_results --csv results/*.csv --outdir figures
python scripts/plot_report_summary.py --result-dir results --outdir figures/report
python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md
python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation
```

Optional real Qwen2.5-VL closed-answer diagnostic sampling:

```bash
python scripts/sample_qwenvl_predictions.py \
  --datasets vqa_rad_closed path_vqa_closed \
  --samples-per-dataset 8 --device cuda \
  --out figures/report/qwenvl_closed_answer_diagnostics.csv
python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation
```

Optional checkpoint-level real Qwen2.5-VL method diagnostics:

```bash
PYTHONPATH=src python scripts/evaluate_qwenvl_checkpoints.py \
  --result-dir results --checkpoint-dir checkpoints \
  --out figures/report/qwenvl_method_prediction_samples.csv \
  --plot-out figures/report/qwenvl_method_validity.png \
  --device cuda --samples-per-run 4
python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation
```

Outputs include:

```text
figures/report/accuracy_mean_std_by_dataset.png
figures/report/async_sync_gap_errorbar.png
figures/report/final_gap_errorbar.png
figures/report/stability_drop_errorbar.png
figures/report/invalid_answer_rate_by_method.png
figures/report/real_qwenvl_vqa_accuracy.png
figures/report/closed_qa_method_comparison.png
figures/report/qwenvl_prediction_validity.png
figures/report/qwenvl_method_validity.png
figures/report/method_rank_mean_std.png
figures/report/fairness_protocol.csv
figures/report/seeded_summary.csv
figures/report/all_seeded_summary.csv
figures/report/primary_seeded_summary_raw.csv
figures/report/delay_stress_summary.csv
figures/report/duplicate_primary_runs.csv
figures/report/mean_std_summary.csv
figures/report/existing_vs_ours_table.csv
figures/report/qwenvl_real_vs_proxy_summary.csv
figures/report/closed_qa_method_comparison.csv
figures/report/qwenvl_closed_answer_diagnostics_summary.csv
figures/report/qwenvl_method_prediction_samples.csv
figures/report/qwenvl_method_validity.csv
presentation/mllm_report_zh.md
presentation/mllm_report_zh.html
```

Important reporting rule: `qwen_vl_proxy` is an earlier compact VQA proxy, while `qwen2_5_vl_3b_qlora` with backend `qwen2_5_vl_3b_4bit_lora_generative` is the real Qwen2.5-VL path. Keep them in separate tables and claims.

## Push Hygiene

Generated outputs are ignored:

```text
results/
figures/
logs/
checkpoints/
data/
__pycache__/
*.egg-info/
```

Commit code, scripts, README, and compact report notes only. Do not force-add downloaded datasets, checkpoints, raw logs, or large figures.
