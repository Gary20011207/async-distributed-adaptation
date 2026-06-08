#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-fedpath-r139}" || true
fi
cd "$ROOT_DIR"
mkdir -p logs results figures/report checkpoints data presentation
STAMP="$(date +%Y%m%d_%H%M%S)"
HEARTBEAT_LOG="logs/mllm_complete24_heartbeat_${STAMP}.log"
ACTION_LOG="logs/mllm_complete24_actions_${STAMP}.log"
QUEUE_FILE="logs/mllm_complete24_queue_${STAMP}.txt"
FAIL_LOG="logs/mllm_complete24_failures_${STAMP}.log"
: > "$HEARTBEAT_LOG"
: > "$ACTION_LOG"
: > "$QUEUE_FILE"
: > "$FAIL_LOG"

CYCLES=144
INTERVAL=600
POLL=30
TEXT_COMMON="--clients 10 --batch-size 4 --local-epochs 1 --lr 0.0002 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.2 --straggler-multiplier 5.0 --eval-every 5 --save-best"
TEXT_ASYNC="--alpha 0.5 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --buffer-size 5"
CAA="--alpha 0.62 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --buffer-size 5 --agreement-epsilon 0.15 --agreement-power 0.5 --delta-clip-multiplier 1.8 --adaptive-alpha-min 0.20 --adaptive-alpha-max 0.70 --adaptive-staleness-scale 10 --server-delta-momentum 0.8 --history-agreement-blend 0.25 --client-fairness-power 0.5"
VQA_PROXY_COMMON="--clients 10 --batch-size 1 --gradient-accumulation-steps 4 --local-epochs 1 --lr 0.0002 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.2 --straggler-multiplier 5.0 --eval-every 5 --save-best"
VL_COMMON="--clients 2 --batch-size 1 --gradient-accumulation-steps 1 --local-epochs 1 --lr 0.00005 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.5 --straggler-multiplier 5.0 --eval-every 1 --max-length 512 --save-best"

log_action() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$ACTION_LOG"
}

enqueue_job() {
  local tag="$1" dataset="$2" model="$3" method="$4" seed="$5" budget="$6" partition="$7" delay="$8" cmd="$9"
  echo "JOB::${tag}::${dataset}::${model}::${method}::${seed}::${budget}::${partition}::${delay}::${cmd}" >> "$QUEUE_FILE"
}

enqueue_action() {
  local tag="$1" cmd="$2"
  echo "ACTION::${tag}::-::-::-::-::-::-::${cmd}" >> "$QUEUE_FILE"
}

summary_exists() {
  local dataset="$1" model="$2" method="$3" seed="$4" budget="$5" partition="$6" delay="$7"
  python - "$dataset" "$model" "$method" "$seed" "$budget" "$partition" "$delay" <<'PY2'
import json
import sys
from pathlib import Path

dataset, model, method, seed, budget, partition, delay = sys.argv[1:]
for path in Path("results").glob("*_summary.json"):
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    config = summary.get("config", {})
    if str(config.get("dataset", "")) != dataset:
        continue
    if str(config.get("model", "")) != model:
        continue
    if str(summary.get("method", config.get("method", ""))) != method:
        continue
    if str(config.get("seed", "")) != seed:
        continue
    if str(config.get("update_budget", summary.get("total_rounds_or_events", ""))) != budget:
        continue
    if str(config.get("partition", "")) != partition:
        continue
    if str(config.get("delay_mode", "")) != delay:
        continue
    raise SystemExit(0)
raise SystemExit(1)
PY2
}

refresh_report() {
  log_action "START refresh_report"
  PYTHONPATH=src python -m fed_mllm.plot_results --csv results/*.csv --outdir figures >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_report_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md >> "$ACTION_LOG" 2>&1 || true
  python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation >> "$ACTION_LOG" 2>&1 || true
  log_action "DONE refresh_report"
}

heartbeat() {
  local cycle="$1"
  local active_pid="${2:-}"
  local active_tag="${3:-}"
  local next_cmd="${4:-}"
  local gpu="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true)"
  fi
  local summary_count="$(find results -name '*_summary.json' 2>/dev/null | wc -l)"
  local csv_count="$(find results -name '*.csv' 2>/dev/null | wc -l)"
  local latest="$(ls -t results/*_summary.json 2>/dev/null | head -n 1 || true)"
  local queue_total="$(wc -l < "$QUEUE_FILE")"
  local fair_summary=""
  if [[ -f figures/report/mean_std_summary.csv ]]; then
    fair_summary="$(python - <<'PY3' 2>/dev/null || true
import pandas as pd
try:
    m=pd.read_csv('figures/report/mean_std_summary.csv')
    f=pd.read_csv('figures/report/fairness_audit.csv')
    print('mean_rows=', len(m), 'fair_groups=', len(f), 'budget_fair=', int(f.get('budget_fair', pd.Series(dtype=bool)).fillna(False).sum()))
except Exception as exc:
    print('report_summary_unavailable=', exc)
PY3
)"
  fi
  {
    echo "===== heartbeat ${cycle}/${CYCLES} $(date --iso-8601=seconds) ====="
    echo "active_pid=${active_pid} active_tag=${active_tag}"
    echo "gpu=${gpu}"
    echo "summary_count=${summary_count} csv_count=${csv_count} queue_index=${QUEUE_INDEX:-1}/${queue_total}"
    echo "latest_summary=${latest}"
    if [[ -n "$latest" ]]; then
      python - "$latest" <<'PY2' 2>/dev/null || true
import json, sys
s=json.load(open(sys.argv[1]))
c=s.get('config', {})
print('latest_dataset=', c.get('dataset'), 'latest_model=', c.get('model'), 'backend=', c.get('model_backend'))
print('latest_method=', s.get('method'), 'best=', s.get('best_test_acc'), 'final=', s.get('final_test_acc'))
print('latest_invalid=', s.get('final_invalid_answer_rate'))
PY2
    fi
    echo "failed_count=$(wc -l < "$FAIL_LOG")"
    echo "fairness=${fair_summary}"
    echo "next_action=${next_cmd}"
  } >> "$HEARTBEAT_LOG"
}

build_queue() {
  enqueue_action "stage0_compile" "python -m py_compile src/fed_mllm/*.py scripts/*.py"
  enqueue_action "stage0_text_smoke" "PYTHONPATH=src python src/fed_mllm/run.py --synthetic --task text_mcqa --dataset mmlu --model tiny_text --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --device cpu --result-dir /tmp/fed_mllm_smoke_results --checkpoint-dir /tmp/fed_mllm_smoke_checkpoints"
  enqueue_action "stage0_vqa_smoke" "PYTHONPATH=src python src/fed_mllm/run.py --synthetic --task medical_vqa --dataset vqa_rad_closed --model tiny_vqa --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --device cpu --result-dir /tmp/fed_mllm_smoke_results --checkpoint-dir /tmp/fed_mllm_smoke_checkpoints"
  enqueue_action "stage0_refresh" "python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation"

  for dataset in medqa_usmle pubmedqa; do
    seed=44
    for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
      if [[ "$method" == "sync_fedavg" ]]; then
        enqueue_job "stage1_text_${dataset}_${method}_s${seed}" "$dataset" "qwen_text_0_5b_lora" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --rounds 5 --seed $seed --max-train-samples 1500 --max-test-samples 500 $TEXT_COMMON"
      elif [[ "$method" == "caa_fedbuff_v2" ]]; then
        enqueue_job "stage1_text_${dataset}_${method}_s${seed}" "$dataset" "qwen_text_0_5b_lora" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 50 --seed $seed --max-train-samples 1500 --max-test-samples 500 $TEXT_COMMON $CAA"
      else
        enqueue_job "stage1_text_${dataset}_${method}_s${seed}" "$dataset" "qwen_text_0_5b_lora" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 50 --seed $seed --max-train-samples 1500 --max-test-samples 500 $TEXT_COMMON $TEXT_ASYNC"
      fi
    done
  done

  for dataset in vqa_rad_closed path_vqa_closed; do
    for seed in 43 44; do
      for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
        if [[ "$method" == "sync_fedavg" ]]; then
          enqueue_job "stage2_proxy_${dataset}_${method}_s${seed}" "$dataset" "qwen_vl_proxy" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --rounds 5 --seed $seed --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON"
        elif [[ "$method" == "caa_fedbuff_v2" ]]; then
          enqueue_job "stage2_proxy_${dataset}_${method}_s${seed}" "$dataset" "qwen_vl_proxy" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed $seed --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON $CAA"
        else
          enqueue_job "stage2_proxy_${dataset}_${method}_s${seed}" "$dataset" "qwen_vl_proxy" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed $seed --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON $TEXT_ASYNC"
        fi
      done
    done
  done

  for dataset in vqa_rad_closed path_vqa_closed; do
    for seed in 43 44; do
      for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
        if [[ "$method" == "sync_fedavg" ]]; then
          enqueue_job "stage3_realvl_${dataset}_${method}_s${seed}" "$dataset" "qwen2_5_vl_3b_qlora" "$method" "$seed" "2" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --rounds 1 --seed $seed --max-train-samples 8 --max-test-samples 8 $VL_COMMON"
        elif [[ "$method" == "caa_fedbuff_v2" ]]; then
          enqueue_job "stage3_realvl_${dataset}_${method}_s${seed}" "$dataset" "qwen2_5_vl_3b_qlora" "$method" "$seed" "2" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --events 2 --seed $seed --max-train-samples 8 --max-test-samples 8 $VL_COMMON $CAA"
        else
          enqueue_job "stage3_realvl_${dataset}_${method}_s${seed}" "$dataset" "qwen2_5_vl_3b_qlora" "$method" "$seed" "2" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --events 2 --seed $seed --max-train-samples 8 --max-test-samples 8 $VL_COMMON $TEXT_ASYNC"
        fi
      done
    done
  done

  for dataset in medmcqa; do
    for delay in uniform lognormal heterogeneous; do
      for method in naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
        extra="$TEXT_ASYNC"
        [[ "$method" == "caa_fedbuff_v2" ]] && extra="$CAA"
        enqueue_job "stage4_stress_${dataset}_${delay}_${method}_s42" "$dataset" "qwen_text_0_5b_lora" "$method" "42" "100" "iid" "$delay" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 100 --seed 42 --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON --delay-mode $delay $extra"
      done
    done
  done

  dataset=vqa_rad_closed
  for delay in uniform lognormal heterogeneous; do
    for method in naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
      extra="$TEXT_ASYNC"
      [[ "$method" == "caa_fedbuff_v2" ]] && extra="$CAA"
      enqueue_job "stage4_stress_proxy_${dataset}_${delay}_${method}_s42" "$dataset" "qwen_vl_proxy" "$method" "42" "50" "iid" "$delay" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed 42 --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON --delay-mode $delay $extra"
    done
  done

  for dataset in medmcqa; do
    for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
      if [[ "$method" == "sync_fedavg" ]]; then
        enqueue_job "stage5_noniid_${dataset}_${method}_s42" "$dataset" "qwen_text_0_5b_lora" "$method" "42" "100" "dirichlet" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --rounds 10 --seed 42 --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON --partition dirichlet --dirichlet-alpha 0.5"
      elif [[ "$method" == "caa_fedbuff_v2" ]]; then
        enqueue_job "stage5_noniid_${dataset}_${method}_s42" "$dataset" "qwen_text_0_5b_lora" "$method" "42" "100" "dirichlet" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 100 --seed 42 --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON --partition dirichlet --dirichlet-alpha 0.5 $CAA"
      else
        enqueue_job "stage5_noniid_${dataset}_${method}_s42" "$dataset" "qwen_text_0_5b_lora" "$method" "42" "100" "dirichlet" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 100 --seed 42 --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON --partition dirichlet --dirichlet-alpha 0.5 $TEXT_ASYNC"
      fi
    done
  done

  dataset=vqa_rad_closed
  for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
    if [[ "$method" == "sync_fedavg" ]]; then
      enqueue_job "stage5_noniid_proxy_${dataset}_${method}_s42" "$dataset" "qwen_vl_proxy" "$method" "42" "50" "dirichlet" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --rounds 5 --seed 42 --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON --partition dirichlet --dirichlet-alpha 0.5"
    elif [[ "$method" == "caa_fedbuff_v2" ]]; then
      enqueue_job "stage5_noniid_proxy_${dataset}_${method}_s42" "$dataset" "qwen_vl_proxy" "$method" "42" "50" "dirichlet" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed 42 --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON --partition dirichlet --dirichlet-alpha 0.5 $CAA"
    else
      enqueue_job "stage5_noniid_proxy_${dataset}_${method}_s42" "$dataset" "qwen_vl_proxy" "$method" "42" "50" "dirichlet" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed 42 --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON --partition dirichlet --dirichlet-alpha 0.5 $TEXT_ASYNC"
    fi
  done

  enqueue_action "stage6_qwenvl_checkpoint_diagnostics" "PYTHONPATH=src python scripts/evaluate_qwenvl_checkpoints.py --result-dir results --checkpoint-dir checkpoints --out figures/report/qwenvl_method_prediction_samples.csv --plot-out figures/report/qwenvl_method_validity.png --device cuda --samples-per-run 4"
  enqueue_action "stage6_final_refresh" "PYTHONPATH=src python -m fed_mllm.plot_results --csv results/*.csv --outdir figures && python scripts/plot_report_summary.py --result-dir results --outdir figures/report && python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md && python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation"
}

run_next_line() {
  local line="$1"
  IFS='::' read -r kind _ tag _ dataset _ model _ method _ seed _ budget _ partition _ delay _ cmd <<< "$line"
  # IFS with repeated ':' is awkward; parse with Python-compatible delimiter fallback below.
  kind="${line%%::*}"
  local rest="${line#*::}"
  tag="${rest%%::*}"; rest="${rest#*::}"
  dataset="${rest%%::*}"; rest="${rest#*::}"
  model="${rest%%::*}"; rest="${rest#*::}"
  method="${rest%%::*}"; rest="${rest#*::}"
  seed="${rest%%::*}"; rest="${rest#*::}"
  budget="${rest%%::*}"; rest="${rest#*::}"
  partition="${rest%%::*}"; rest="${rest#*::}"
  delay="${rest%%::*}"; rest="${rest#*::}"
  cmd="$rest"
  if [[ "$kind" == "JOB" ]] && summary_exists "$dataset" "$model" "$method" "$seed" "$budget" "$partition" "$delay"; then
    log_action "SKIP $tag existing_summary dataset=$dataset model=$model method=$method seed=$seed budget=$budget partition=$partition delay=$delay"
    ACTIVE_PID=""
    ACTIVE_TAG=""
    ACTIVE_LOG=""
    return 0
  fi
  local run_log="logs/${tag}_${STAMP}.log"
  log_action "START $tag"
  bash -lc "$cmd" > "$run_log" 2>&1 &
  ACTIVE_PID="$!"
  ACTIVE_TAG="$tag"
  ACTIVE_LOG="$run_log"
}

build_queue
log_action "queue_size=$(wc -l < "$QUEUE_FILE")"
CYCLE=1
QUEUE_INDEX=1
ACTIVE_PID=""
ACTIVE_TAG=""
ACTIVE_LOG=""
NEXT_CMD="$(sed -n "${QUEUE_INDEX}p" "$QUEUE_FILE")"
LAST_HEARTBEAT=0

while [[ $CYCLE -le $CYCLES ]]; do
  now=$(date +%s)
  if [[ -z "$ACTIVE_PID" && -n "$NEXT_CMD" ]]; then
    ACTIVE_PID=""; ACTIVE_TAG=""; ACTIVE_LOG=""
    run_next_line "$NEXT_CMD"
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
    NEXT_CMD="$(sed -n "${QUEUE_INDEX}p" "$QUEUE_FILE")"
  fi

  if [[ -n "$ACTIVE_PID" ]] && ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
    wait "$ACTIVE_PID"
    status=$?
    if [[ $status -ne 0 ]]; then
      echo "$(date --iso-8601=seconds) $ACTIVE_TAG status=$status log=$ACTIVE_LOG" >> "$FAIL_LOG"
      log_action "FAIL $ACTIVE_TAG status=$status log=$ACTIVE_LOG"
    else
      log_action "DONE $ACTIVE_TAG"
    fi
    ACTIVE_PID=""; ACTIVE_TAG=""; ACTIVE_LOG=""
    refresh_report
  fi

  if [[ $LAST_HEARTBEAT -eq 0 || $((now - LAST_HEARTBEAT)) -ge $INTERVAL ]]; then
    heartbeat "$CYCLE" "$ACTIVE_PID" "$ACTIVE_TAG" "$NEXT_CMD"
    CYCLE=$((CYCLE + 1))
    LAST_HEARTBEAT=$now
  fi
  sleep "$POLL"
done

if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
  log_action "WAIT_ACTIVE_AFTER_HEARTBEATS $ACTIVE_TAG pid=$ACTIVE_PID"
  wait "$ACTIVE_PID"
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "$(date --iso-8601=seconds) $ACTIVE_TAG status=$status log=$ACTIVE_LOG" >> "$FAIL_LOG"
    log_action "FAIL $ACTIVE_TAG status=$status log=$ACTIVE_LOG"
  else
    log_action "DONE $ACTIVE_TAG"
  fi
fi

log_action "heartbeat completed cycles=$CYCLES"
refresh_report
