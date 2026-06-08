#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-fedpath-r139}" || true
fi
cd "$ROOT_DIR"
mkdir -p logs results figures/report checkpoints data
STAMP="$(date +%Y%m%d_%H%M%S)"
HEARTBEAT_LOG="logs/qwenvl12_heartbeat_${STAMP}.log"
ACTION_LOG="logs/qwenvl12_actions_${STAMP}.log"
QUEUE_FILE="logs/qwenvl12_queue_${STAMP}.txt"
FAIL_LOG="logs/qwenvl12_failures_${STAMP}.log"
: > "$HEARTBEAT_LOG"
: > "$ACTION_LOG"
: > "$QUEUE_FILE"
: > "$FAIL_LOG"

CYCLES=72
INTERVAL=600
POLL=30
TEXT_COMMON="--clients 10 --batch-size 4 --local-epochs 1 --lr 0.0002 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.2 --straggler-multiplier 5.0 --eval-every 5"
TEXT_ASYNC="--alpha 0.5 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --buffer-size 5"
CAA="--alpha 0.62 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --buffer-size 5 --agreement-epsilon 0.15 --agreement-power 0.5 --delta-clip-multiplier 1.8 --adaptive-alpha-min 0.20 --adaptive-alpha-max 0.70 --adaptive-staleness-scale 10 --server-delta-momentum 0.8 --history-agreement-blend 0.25 --client-fairness-power 0.5"
VL_COMMON="--clients 2 --batch-size 1 --local-epochs 1 --lr 0.00005 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.5 --straggler-multiplier 5.0 --eval-every 1 --max-length 512 --gradient-accumulation-steps 1"

log_action() { echo "[$(date --iso-8601=seconds)] $*" | tee -a "$ACTION_LOG"; }
enqueue() { echo "$*" >> "$QUEUE_FILE"; }

refresh_report() {
  PYTHONPATH=src python -m fed_mllm.plot_results --csv 'results/*.csv' --outdir figures >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_report_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md >> "$ACTION_LOG" 2>&1 || true
}

heartbeat() {
  local cycle="$1"
  local active_pid="${2:-}"
  local next_cmd="${3:-}"
  local gpu="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true)"
  fi
  local summary_count="$(find results -name '*_summary.json' 2>/dev/null | wc -l)"
  local latest="$(ls -t results/*_summary.json 2>/dev/null | head -n 1 || true)"
  {
    echo "===== heartbeat ${cycle}/${CYCLES} $(date --iso-8601=seconds) ====="
    echo "active_pid=${active_pid}"
    echo "gpu=${gpu}"
    echo "summary_count=${summary_count}"
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
    echo "next_action=${next_cmd}"
  } >> "$HEARTBEAT_LOG"
}

build_queue() {
  enqueue "stage0_compile::python -m py_compile src/fed_mllm/*.py scripts/*.py"
  enqueue "stage0_qwenvl_synthetic_smoke::PYTHONPATH=src python src/fed_mllm/run.py --synthetic --task medical_vqa --dataset vqa_rad_closed --model qwen2_5_vl_3b_qlora --method sync_fedavg --rounds 1 --clients 1 --batch-size 1 --max-train-samples 1 --max-test-samples 1 --max-length 512 --device cuda --result-dir /tmp/fed_mllm_qwenvl_smoke_results --checkpoint-dir /tmp/fed_mllm_qwenvl_smoke_checkpoints"

  for dataset in medqa_usmle pubmedqa; do
    for seed in 42 43; do
      for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
        if [[ "$method" == "sync_fedavg" ]]; then
          enqueue "stage1_text_${dataset}_${method}_s${seed}::PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --rounds 5 --seed $seed --max-train-samples 1500 --max-test-samples 500 $TEXT_COMMON --save-best"
        elif [[ "$method" == "caa_fedbuff_v2" ]]; then
          enqueue "stage1_text_${dataset}_${method}_s${seed}::PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 50 --seed $seed --max-train-samples 1500 --max-test-samples 500 $TEXT_COMMON $CAA --save-best"
        else
          enqueue "stage1_text_${dataset}_${method}_s${seed}::PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 50 --seed $seed --max-train-samples 1500 --max-test-samples 500 $TEXT_COMMON $TEXT_ASYNC --save-best"
        fi
      done
    done
  done

  for dataset in vqa_rad_closed path_vqa_closed; do
    for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
      if [[ "$method" == "sync_fedavg" ]]; then
        enqueue "stage2_realvl_${dataset}_${method}_s42::PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --rounds 1 --seed 42 --max-train-samples 8 --max-test-samples 8 $VL_COMMON"
      elif [[ "$method" == "caa_fedbuff_v2" ]]; then
        enqueue "stage2_realvl_${dataset}_${method}_s42::PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --events 2 --seed 42 --max-train-samples 8 --max-test-samples 8 $VL_COMMON $CAA"
      else
        enqueue "stage2_realvl_${dataset}_${method}_s42::PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --events 2 --seed 42 --max-train-samples 8 --max-test-samples 8 $VL_COMMON $TEXT_ASYNC"
      fi
    done
  done
}

run_next_command() {
  local line="$1"
  local tag="${line%%::*}"
  local cmd="${line#*::}"
  local run_log="logs/${tag}_${STAMP}.log"
  log_action "START $tag"
  bash -lc "$cmd" > "$run_log" 2>&1 &
  ACTIVE_PID="$!"
  ACTIVE_TAG="$tag"
  ACTIVE_LOG="$run_log"
}

build_queue
log_action "queue_size=$(wc -l < "$QUEUE_FILE")"
cycle=1
queue_index=1
active_pid=""
active_tag=""
active_log=""
next_cmd="$(sed -n "${queue_index}p" "$QUEUE_FILE")"
last_heartbeat=0

while [[ $cycle -le $CYCLES ]]; do
  now=$(date +%s)
  if [[ -z "$active_pid" && -n "$next_cmd" ]]; then
    ACTIVE_PID=""; ACTIVE_TAG=""; ACTIVE_LOG=""
    run_next_command "$next_cmd"
    active_pid="$ACTIVE_PID"; active_tag="$ACTIVE_TAG"; active_log="$ACTIVE_LOG"
    queue_index=$((queue_index + 1))
    next_cmd="$(sed -n "${queue_index}p" "$QUEUE_FILE")"
  fi

  if [[ -n "$active_pid" ]] && ! kill -0 "$active_pid" 2>/dev/null; then
    wait "$active_pid"
    status=$?
    if [[ $status -ne 0 ]]; then
      echo "$(date --iso-8601=seconds) $active_tag status=$status log=$active_log" >> "$FAIL_LOG"
      log_action "FAIL $active_tag status=$status"
    else
      log_action "DONE $active_tag"
    fi
    active_pid=""; active_tag=""; active_log=""
    refresh_report
  fi

  if [[ $last_heartbeat -eq 0 || $((now - last_heartbeat)) -ge $INTERVAL ]]; then
    heartbeat "$cycle" "$active_pid" "$next_cmd"
    cycle=$((cycle + 1))
    last_heartbeat=$now
  fi
  sleep "$POLL"
done

log_action "heartbeat completed cycles=$CYCLES"
refresh_report
