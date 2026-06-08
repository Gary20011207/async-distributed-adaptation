#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-fedpath-r139}" || true
fi
cd "$ROOT_DIR"
mkdir -p logs results figures/report checkpoints presentation

STAMP="$(date +%Y%m%d_%H%M%S)"
HEARTBEAT_LOG="logs/mllm_halfday_heartbeat_${STAMP}.log"
ACTION_LOG="logs/mllm_halfday_actions_${STAMP}.log"
FAIL_LOG="logs/mllm_halfday_failures_${STAMP}.log"
QUEUE_FILE="logs/mllm_halfday_queue_${STAMP}.txt"
: > "$HEARTBEAT_LOG"
: > "$ACTION_LOG"
: > "$FAIL_LOG"
: > "$QUEUE_FILE"

CYCLES="${CYCLES:-72}"
INTERVAL="${INTERVAL:-600}"
POLL="${POLL:-30}"

TEXT_COMMON="--clients 10 --batch-size 4 --local-epochs 1 --lr 0.0002 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.2 --straggler-multiplier 5.0 --eval-every 5 --save-best"
TEXT_ASYNC="--alpha 0.5 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --buffer-size 5"
CAA="--alpha 0.62 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05 --buffer-size 5 --agreement-epsilon 0.15 --agreement-power 0.5 --delta-clip-multiplier 1.8 --adaptive-alpha-min 0.20 --adaptive-alpha-max 0.70 --adaptive-staleness-scale 10 --server-delta-momentum 0.8 --history-agreement-blend 0.25 --client-fairness-power 0.5"
VQA_PROXY_COMMON="--clients 10 --batch-size 1 --gradient-accumulation-steps 4 --local-epochs 1 --lr 0.0002 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.2 --straggler-multiplier 5.0 --eval-every 5 --save-best"
VL_COMMON="--clients 2 --batch-size 1 --gradient-accumulation-steps 1 --local-epochs 1 --lr 0.00005 --partition iid --device cuda --delay-mode heterogeneous --straggler-ratio 0.5 --straggler-multiplier 5.0 --eval-every 1 --max-length 512 --save-best"

log_action() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$ACTION_LOG"
}

cuda_available() {
  python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
}

summary_exists() {
  local result_dir="$1" dataset="$2" model="$3" method="$4" seed="$5" budget="$6" partition="$7" delay="$8"
  python - "$result_dir" "$dataset" "$model" "$method" "$seed" "$budget" "$partition" "$delay" <<'PY'
import json
import sys
from pathlib import Path

result_dir, dataset, model, method, seed, budget, partition, delay = sys.argv[1:]
for path in Path(result_dir).glob("*_summary.json"):
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
PY
}

enqueue() {
  local tag="$1" result_dir="$2" dataset="$3" model="$4" method="$5" seed="$6" budget="$7" partition="$8" delay="$9"
  shift 9
  local cmd="$*"
  echo "${tag}::${result_dir}::${dataset}::${model}::${method}::${seed}::${budget}::${partition}::${delay}::${cmd}" >> "$QUEUE_FILE"
}

refresh_report() {
  log_action "START refresh_report"
  PYTHONPATH=src python -m fed_mllm.plot_results --csv results/*.csv --outdir figures >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_report_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md >> "$ACTION_LOG" 2>&1 || true
  python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation >> "$ACTION_LOG" 2>&1 || true
  python scripts/create_mllm_demo_deck.py >> "$ACTION_LOG" 2>&1 || true
  log_action "DONE refresh_report"
}

heartbeat() {
  local cycle="$1" active_pid="${2:-}" active_tag="${3:-}" next_line="${4:-}"
  local gpu="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true)"
  fi
  local cuda_state="no"
  cuda_available && cuda_state="yes"
  local latest="$(ls -t results/*_summary.json 2>/dev/null | head -n 1 || true)"
  {
    echo "===== heartbeat ${cycle}/${CYCLES} $(date --iso-8601=seconds) ====="
    echo "cuda_available=${cuda_state}"
    echo "active_pid=${active_pid} active_tag=${active_tag}"
    echo "gpu=${gpu}"
    echo "queue_index=${QUEUE_INDEX:-1}/$(wc -l < "$QUEUE_FILE")"
    echo "summary_count=$(find results -name '*_summary.json' 2>/dev/null | wc -l)"
    echo "demo_summary_count=$(find results_demo_halfday -name '*_summary.json' 2>/dev/null | wc -l)"
    echo "failed_count=$(wc -l < "$FAIL_LOG")"
    echo "latest_summary=${latest}"
    if [[ -n "$latest" ]]; then
      python - "$latest" <<'PY' 2>/dev/null || true
import json, sys
s=json.load(open(sys.argv[1]))
c=s.get("config", {})
print("latest=", c.get("dataset"), c.get("model"), s.get("method"), "best=", s.get("best_test_acc"), "final=", s.get("final_test_acc"))
PY
    fi
    echo "next=${next_line}"
  } >> "$HEARTBEAT_LOG"
}

build_queue() {
  # Stage 0: report refresh and diagnostics do not change official experimental claims.
  enqueue "stage0_compile" "-" "-" "-" "-" "-" "-" "-" "-" "python -m py_compile src/fed_mllm/*.py scripts/*.py"
  enqueue "stage0_cpu_smoke" "-" "-" "-" "-" "-" "-" "-" "-" "PYTHONPATH=src python src/fed_mllm/run.py --synthetic --task text_mcqa --dataset mmlu --model tiny_text --method caa_fedbuff_v2 --events 4 --clients 2 --buffer-size 2 --device cpu --result-dir /tmp/fed_mllm_halfday_smoke_results --checkpoint-dir /tmp/fed_mllm_halfday_smoke_checkpoints"
  enqueue "stage0_refresh" "-" "-" "-" "-" "-" "-" "-" "-" "python scripts/create_mllm_demo_deck.py"

  # Stage 1: add one more fair seed for strongest text MCQA demo datasets.
  for dataset in mmlu medmcqa; do
    seed=45
    for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
      if [[ "$method" == "sync_fedavg" ]]; then
        enqueue "stage1_text_${dataset}_${method}_s${seed}" "results" "$dataset" "qwen_text_0_5b_lora" "$method" "$seed" "100" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --rounds 10 --seed $seed --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON"
      elif [[ "$method" == "caa_fedbuff_v2" ]]; then
        enqueue "stage1_text_${dataset}_${method}_s${seed}" "results" "$dataset" "qwen_text_0_5b_lora" "$method" "$seed" "100" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 100 --seed $seed --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON $CAA"
      else
        enqueue "stage1_text_${dataset}_${method}_s${seed}" "results" "$dataset" "qwen_text_0_5b_lora" "$method" "$seed" "100" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task text_mcqa --dataset $dataset --model qwen_text_0_5b_lora --method $method --events 100 --seed $seed --max-train-samples 3000 --max-test-samples 1000 $TEXT_COMMON $TEXT_ASYNC"
      fi
    done
  done

  # Stage 2: add one more fair seed for proxy VQA, which is fast enough for demo comparison.
  for dataset in vqa_rad_closed path_vqa_closed; do
    seed=45
    for method in sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
      if [[ "$method" == "sync_fedavg" ]]; then
        enqueue "stage2_proxy_${dataset}_${method}_s${seed}" "results" "$dataset" "qwen_vl_proxy" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --rounds 5 --seed $seed --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON"
      elif [[ "$method" == "caa_fedbuff_v2" ]]; then
        enqueue "stage2_proxy_${dataset}_${method}_s${seed}" "results" "$dataset" "qwen_vl_proxy" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed $seed --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON $CAA"
      else
        enqueue "stage2_proxy_${dataset}_${method}_s${seed}" "results" "$dataset" "qwen_vl_proxy" "$method" "$seed" "50" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen_vl_proxy --method $method --events 50 --seed $seed --max-train-samples 1000 --max-test-samples 300 $VQA_PROXY_COMMON $TEXT_ASYNC"
      fi
    done
  done

  # Stage 3: keep real Qwen2.5-VL heavier confirmation separate from headline results.
  mkdir -p results_demo_halfday checkpoints_demo_halfday
  dataset=vqa_rad_closed
  for method in sync_fedavg naive_async caa_fedbuff_v2; do
    if [[ "$method" == "sync_fedavg" ]]; then
      enqueue "stage3_realvl_demo_${dataset}_${method}_s45" "results_demo_halfday" "$dataset" "qwen2_5_vl_3b_qlora" "$method" "45" "4" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --rounds 2 --seed 45 --max-train-samples 16 --max-test-samples 16 --result-dir results_demo_halfday --checkpoint-dir checkpoints_demo_halfday $VL_COMMON"
    elif [[ "$method" == "caa_fedbuff_v2" ]]; then
      enqueue "stage3_realvl_demo_${dataset}_${method}_s45" "results_demo_halfday" "$dataset" "qwen2_5_vl_3b_qlora" "$method" "45" "4" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --events 4 --seed 45 --max-train-samples 16 --max-test-samples 16 --result-dir results_demo_halfday --checkpoint-dir checkpoints_demo_halfday $VL_COMMON $CAA"
    else
      enqueue "stage3_realvl_demo_${dataset}_${method}_s45" "results_demo_halfday" "$dataset" "qwen2_5_vl_3b_qlora" "$method" "45" "4" "iid" "heterogeneous" "PYTHONPATH=src python src/fed_mllm/run.py --task medical_vqa --dataset $dataset --model qwen2_5_vl_3b_qlora --method $method --events 4 --seed 45 --max-train-samples 16 --max-test-samples 16 --result-dir results_demo_halfday --checkpoint-dir checkpoints_demo_halfday $VL_COMMON $TEXT_ASYNC"
    fi
  done

  enqueue "stage4_checkpoint_diagnostics" "-" "-" "-" "-" "-" "-" "-" "-" "PYTHONPATH=src python scripts/evaluate_qwenvl_checkpoints.py --result-dir results --checkpoint-dir checkpoints --out figures/report/qwenvl_method_prediction_samples.csv --plot-out figures/report/qwenvl_method_validity.png --device cuda --samples-per-run 8"
  enqueue "stage5_final_refresh" "-" "-" "-" "-" "-" "-" "-" "-" "PYTHONPATH=src python -m fed_mllm.plot_results --csv results/*.csv --outdir figures && python scripts/plot_report_summary.py --result-dir results --outdir figures/report && python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md && python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation && python scripts/create_mllm_demo_deck.py"
}

parse_line() {
  local line="$1"
  tag="${line%%::*}"; rest="${line#*::}"
  result_dir="${rest%%::*}"; rest="${rest#*::}"
  dataset="${rest%%::*}"; rest="${rest#*::}"
  model="${rest%%::*}"; rest="${rest#*::}"
  method="${rest%%::*}"; rest="${rest#*::}"
  seed="${rest%%::*}"; rest="${rest#*::}"
  budget="${rest%%::*}"; rest="${rest#*::}"
  partition="${rest%%::*}"; rest="${rest#*::}"
  delay="${rest%%::*}"; rest="${rest#*::}"
  cmd="$rest"
}

start_line() {
  local line="$1"
  parse_line "$line"
  if [[ "$result_dir" != "-" ]] && summary_exists "$result_dir" "$dataset" "$model" "$method" "$seed" "$budget" "$partition" "$delay"; then
    log_action "SKIP $tag existing_summary result_dir=$result_dir dataset=$dataset model=$model method=$method seed=$seed budget=$budget"
    return 1
  fi
  if [[ "$cmd" == *"--device cuda"* ]] && ! cuda_available; then
    return 2
  fi
  local run_log="logs/${tag}_${STAMP}.log"
  log_action "START $tag"
  bash -lc "$cmd" > "$run_log" 2>&1 &
  ACTIVE_PID="$!"
  ACTIVE_TAG="$tag"
  ACTIVE_LOG="$run_log"
  return 0
}

build_queue
log_action "halfday demo optimization queue_size=$(wc -l < "$QUEUE_FILE")"

ACTIVE_PID=""
ACTIVE_TAG=""
ACTIVE_LOG=""
QUEUE_INDEX=1
CYCLE=1
LAST_HEARTBEAT=0
LAST_REFRESH="$(date +%s)"

while [[ $CYCLE -le $CYCLES ]]; do
  now="$(date +%s)"
  next_line="$(sed -n "${QUEUE_INDEX}p" "$QUEUE_FILE")"

  if [[ -z "$ACTIVE_PID" && -n "$next_line" ]]; then
    start_line "$next_line"
    status=$?
    if [[ $status -eq 0 || $status -eq 1 ]]; then
      QUEUE_INDEX=$((QUEUE_INDEX + 1))
    elif [[ $status -eq 2 ]]; then
      log_action "WAIT_CUDA for $(echo "$next_line" | cut -d: -f1)"
    fi
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
    ACTIVE_PID=""
    ACTIVE_TAG=""
    ACTIVE_LOG=""
    refresh_report
  fi

  if [[ $LAST_HEARTBEAT -eq 0 || $((now - LAST_HEARTBEAT)) -ge $INTERVAL ]]; then
    heartbeat "$CYCLE" "$ACTIVE_PID" "$ACTIVE_TAG" "$next_line"
    CYCLE=$((CYCLE + 1))
    LAST_HEARTBEAT=$now
  fi

  if [[ -z "$ACTIVE_PID" && $((now - LAST_REFRESH)) -ge 3600 ]]; then
    refresh_report
    LAST_REFRESH=$now
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
refresh_report
log_action "halfday demo optimization completed cycles=$CYCLES queue_index=$QUEUE_INDEX"
