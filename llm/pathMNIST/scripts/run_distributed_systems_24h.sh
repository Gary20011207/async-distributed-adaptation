#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

HEARTBEAT_LOG="$LOG_DIR/distsys24_heartbeat_${STAMP}.log"
ACTION_LOG="$LOG_DIR/distsys24_actions_${STAMP}.log"
MARKER="$LOG_DIR/distsys24_${STAMP}.marker"
touch "$MARKER"

CYCLES="${CYCLES:-144}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
POLL_SECONDS="${POLL_SECONDS:-30}"
IDLE_REPORT_SECONDS="${IDLE_REPORT_SECONDS:-3600}"

COMMON_ARGS=(
  --clients 10
  --batch-size 4          
  --lr 0.0001 
  --lr-scheduler cosine
  --min-lr 0.00001
  --local-epochs 1
  --max-train-samples 10000
  --max-test-samples 2000
  --device cuda
  --save-best
)

ASYNC_ARGS=(
  --eval-every 20
)

CAA_ARGS=(
  --buffer-size 5
  --alpha 0.62
  --staleness-decay hinge
  --staleness-hinge-b 5
  --staleness-hinge-a 0.05
  --agreement-epsilon 0.15
  --agreement-power 0.5
  --agreement-drop-threshold -0.05
  --delta-clip-multiplier 1.8
  --adaptive-alpha-min 0.20
  --adaptive-alpha-max 0.70
  --adaptive-alpha-boost 0.25
  --adaptive-staleness-scale 10
)

actions=()
failed_runs=()

log_action() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$ACTION_LOG"
}

add_action() {
  actions+=("$*")
}

dataset_extra_args() {
  local dataset="$1"
  case "$dataset" in
    *)
      echo ""
      ;;
  esac
}

rounds_for_dataset() {
  local dataset="$1"
  if [[ "$dataset" == "mmlu" ]]; then
    echo 10
  else
    echo 30
  fi
}

delay_args_for_key() {
  local delay_key="$1"
  case "$delay_key" in
    uniform)
      echo "--delay-mode uniform --min-delay 1.0 --max-delay 10.0"
      ;;
    lognormal)
      echo "--delay-mode lognormal --lognormal-mean 1.0 --lognormal-sigma 0.5"
      ;;
    heterogeneous_mild)
      echo "--delay-mode heterogeneous --straggler-ratio 0.2 --straggler-multiplier 5.0"
      ;;
    heterogeneous_severe)
      echo "--delay-mode heterogeneous --straggler-ratio 0.4 --straggler-multiplier 8.0"
      ;;
    none)
      echo ""
      ;;
    *)
      echo "$delay_key"
      ;;
  esac
}

delay_signature_for_key() {
  local delay_key="$1"
  case "$delay_key" in
    uniform)
      echo "delay_mode=uniform,min_delay=1.0,max_delay=10.0"
      ;;
    lognormal)
      echo "delay_mode=lognormal,lognormal_mean=1.0,lognormal_sigma=0.5"
      ;;
    heterogeneous_mild)
      echo "delay_mode=heterogeneous,straggler_ratio=0.2,straggler_multiplier=5.0"
      ;;
    heterogeneous_severe)
      echo "delay_mode=heterogeneous,straggler_ratio=0.4,straggler_multiplier=8.0"
      ;;
    *)
      echo ""
      ;;
  esac
}

append_signature() {
  local left="${1:-}"
  local right="${2:-}"
  if [[ -z "$left" ]]; then
    echo "$right"
  elif [[ -z "$right" ]]; then
    echo "$left"
  else
    echo "${left},${right}"
  fi
}

enqueue_run() {
  local stage="$1"
  local dataset="$2"
  local method="$3"
  local seed="$4"
  local model="$5"
  local partition="$6"
  local dirichlet_alpha="$7"
  local delay_key="$8"
  local label="$9"
  local signature="${10:-}"
  local variant="${11:-default}"
  local rounds
  local events
  local budget
  local extra
  local partition_signature=""
  local delay_signature=""

  rounds="$(rounds_for_dataset "$dataset")"
  events=$((rounds * 10))
  budget="$events"
  extra="$(dataset_extra_args "$dataset")"

  if [[ "$partition" == "dirichlet" ]]; then
    partition_signature="partition=dirichlet,dirichlet_alpha=${dirichlet_alpha}"
  elif [[ "$partition" == "iid" ]]; then
    partition_signature="partition=iid"
  fi
  if [[ "$method" != "sync_fedavg" ]]; then
    delay_signature="$(delay_signature_for_key "$delay_key")"
  fi
  signature="$(append_signature "$signature" "$partition_signature")"
  signature="$(append_signature "$signature" "$delay_signature")"

  add_action "$stage|$dataset|$method|$seed|$model|$partition|$dirichlet_alpha|$delay_key|$rounds|$events|$budget|$label|$extra|$signature|$variant"
}

# build_queue() {
#   local dataset
#   local method
#   local seed
#   local alpha
#   local delay
#   local variant
#   local ablation_datasets=(pathmnist bloodmnist organamnist)
#   local methods=(sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2)

#   # Stage 1A: seed-42 non-IID hospital scenario first.
#   for dataset in pathmnist pneumoniamnist bloodmnist organamnist; do
#     for alpha in 0.5 0.1; do
#       for method in "${methods[@]}"; do
#         delay="none"
#         [[ "$method" != "sync_fedavg" ]] && delay="heterogeneous_mild"
#         enqueue_run "stage1_noniid_seed42" "$dataset" "$method" 42 resnet18 dirichlet "$alpha" "$delay" "noniid_${dataset}_a${alpha}_${method}_s42"
#       done
#     done
#   done

#   # Stage 2: delay stress makes the distributed timing issue visible.
#   for dataset in pathmnist bloodmnist organamnist; do
#     for delay in uniform lognormal heterogeneous_mild heterogeneous_severe; do
#       for method in naive_async staleness_async fedbuff_async caa_fedbuff_v2; do
#         enqueue_run "stage2_straggler_stress" "$dataset" "$method" 42 resnet18 iid 0.5 "$delay" "stress_${dataset}_${delay}_${method}_s42"
#       done
#     done
#   done

#   # Stage 3A: CAA-v2 ablation, PathMNIST gets multi-seed priority.
#   for seed in 42 43 44; do
#     enqueue_ablation pathmnist "$seed"
#   done
#   # Stage 3B: representative non-Path datasets, seed 42 first.
#   for dataset in bloodmnist organamnist; do
#     enqueue_ablation "$dataset" 42
#   done

#   # Stage 1B: extra non-IID seeds if time remains.
#   for seed in 43 44; do
#     for dataset in pathmnist pneumoniamnist bloodmnist organamnist; do
#       for alpha in 0.5 0.1; do
#         for method in "${methods[@]}"; do
#           delay="none"
#           [[ "$method" != "sync_fedavg" ]] && delay="heterogeneous_mild"
#           enqueue_run "stage1_noniid_extra_seeds" "$dataset" "$method" "$seed" resnet18 dirichlet "$alpha" "$delay" "noniid_${dataset}_a${alpha}_${method}_s${seed}"
#         done
#       done
#     done
#   done

#   # Stage 3C: extra ablation seeds if there is still time.
#   for seed in 43 44; do
#     for dataset in "${ablation_datasets[@]}"; do
#       [[ "$dataset" == "pathmnist" ]] && continue
#       enqueue_ablation "$dataset" "$seed"
#     done
#   done
# }
build_queue() {
  local dataset="mmlu"
  local model="qwen"
  local method
  local seed
  local alpha
  local delay
  local variant
  local methods=(sync_fedavg naive_async staleness_async fedbuff_async caa_fedbuff_v2)

  # 🏥 Stage 1: Qwen 在非獨立同分布（Non-IID）下的表現（多種子測試）
  for seed in 42 43 44; do
    for alpha in 0.5 0.1; do
      for method in "${methods[@]}"; do
        delay="none"
        [[ "$method" != "sync_fedavg" ]] && delay="heterogeneous_mild"
        
        enqueue_run "stage1_qwen_noniid" "$dataset" "$method" "$seed" "$model" dirichlet "$alpha" "$delay" "qwen_noniid_${dataset}_a${alpha}_${method}_s${seed}"
      done
    done
  done

  # ⚡ Stage 2: 異構網路環境隨機延遲「壓力測試」
  for delay in uniform lognormal heterogeneous_mild heterogeneous_severe; do
    for method in "${methods[@]}"; do
      enqueue_run "stage2_qwen_stress" "$dataset" "$method" 42 "$model" iid 0.5 "$delay" "qwen_stress_${dataset}_${delay}_${method}_s42"
    done
  done

  # 🧬 Stage 3: CAA-v2 核心演算法消融實驗（Ablation Study）
  for seed in 42 43; do
    enqueue_run "stage3_qwen_ablation" "$dataset" caa_fedbuff_v2 "$seed" "$model" iid 0.5 heterogeneous_mild "qwen_ablation_${dataset}_full_caa_v2_s${seed}" "history_agreement_blend=0.25,client_fairness_power=0.5,delta_clip_multiplier=1.8" "full_caa_v2"
    enqueue_run "stage3_qwen_ablation" "$dataset" caa_fedbuff_v2 "$seed" "$model" iid 0.5 heterogeneous_mild "qwen_ablation_${dataset}_no_server_ema_s${seed}" "history_agreement_blend=0.0,client_fairness_power=0.5,delta_clip_multiplier=1.8" "no_server_ema"
    enqueue_run "stage3_qwen_ablation" "$dataset" caa_fedbuff_v2 "$seed" "$model" iid 0.5 heterogeneous_mild "qwen_ablation_${dataset}_no_fairness_s${seed}" "history_agreement_blend=0.25,client_fairness_power=0.0,delta_clip_multiplier=1.8" "no_fairness"
    enqueue_run "stage3_qwen_ablation" "$dataset" caa_fedbuff_v2 "$seed" "$model" iid 0.5 heterogeneous_mild "qwen_ablation_${dataset}_no_clipping_s${seed}" "history_agreement_blend=0.25,client_fairness_power=0.5,delta_clip_multiplier=1000000.0" "no_clipping"
    
    local sig_static="history_agreement_blend=0.25,client_fairness_power=0.5,adaptive_alpha_min=0.62,adaptive_alpha_max=0.62"
    enqueue_run "stage3_qwen_ablation" "$dataset" caa_fedbuff_v2 "$seed" "$model" iid 0.5 heterogeneous_mild "qwen_ablation_${dataset}_static_alpha_s${seed}" "$sig_static" "static_alpha"
  done
}

enqueue_ablation() {
  local dataset="$1"
  local seed="$2"
  local signature

  enqueue_run "stage3_caa_v2_ablation" "$dataset" caa_fedbuff_v2 "$seed" qwen iid 0.5 heterogeneous_mild "ablation_${dataset}_full_caa_v2_s${seed}" "history_agreement_blend=0.25,client_fairness_power=0.5,delta_clip_multiplier=1.8" "full_caa_v2"
  enqueue_run "stage3_caa_v2_ablation" "$dataset" caa_fedbuff_v2 "$seed" qwen iid 0.5 heterogeneous_mild "ablation_${dataset}_no_server_ema_s${seed}" "history_agreement_blend=0.0,client_fairness_power=0.5,delta_clip_multiplier=1.8" "no_server_ema"
  enqueue_run "stage3_caa_v2_ablation" "$dataset" caa_fedbuff_v2 "$seed" qwen iid 0.5 heterogeneous_mild "ablation_${dataset}_no_fairness_s${seed}" "history_agreement_blend=0.25,client_fairness_power=0.0,delta_clip_multiplier=1.8" "no_fairness"
  enqueue_run "stage3_caa_v2_ablation" "$dataset" caa_fedbuff_v2 "$seed" qwen iid 0.5 heterogeneous_mild "ablation_${dataset}_no_clipping_s${seed}" "history_agreement_blend=0.25,client_fairness_power=0.5,delta_clip_multiplier=1000000.0" "no_clipping"
  signature="history_agreement_blend=0.25,client_fairness_power=0.5,adaptive_alpha_min=0.62,adaptive_alpha_max=0.62"
  enqueue_run "stage3_caa_v2_ablation" "$dataset" caa_fedbuff_v2 "$seed" qwen iid 0.5 heterogeneous_mild "ablation_${dataset}_static_alpha_s${seed}" "$signature" "static_alpha"
  enqueue_run "stage3_caa_v2_ablation" "$dataset" agreement_fedbuff_async "$seed" qwen iid 0.5 heterogeneous_mild "ablation_${dataset}_old_caa_s${seed}" "" "old_caa"
}

active_training_commands() {
  ps -ef | grep 'src/fed_pathmnist/run.py' | grep -v grep || true
}

gpu_status() {
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}

summary_exists() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local model="$4"
  local budget="$5"
  local signature="${6:-}"
  python - "$dataset" "$method" "$seed" "$model" "$budget" "$signature" <<'PY'
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

dataset, method, seed, model, budget, signature = sys.argv[1:7]
seed = int(seed)
budget = float(budget)


def dataset_of(path: str, summary: dict) -> str:
    config = summary.get("config", {})
    if config.get("dataset"):
        return str(config["dataset"])
    method_name = summary.get("method", "")
    name = Path(path).name
    prefix = f"{method_name}_"
    if name.startswith(prefix):
        parts = name[len(prefix):].split("_")
        if len(parts) >= 4:
            return "_".join(parts[:-3])
    return "pathmnist"


def update_budget(summary: dict, config: dict) -> float:
    if config.get("update_budget") not in (None, ""):
        return float(config["update_budget"])
    progress = float(summary.get("total_rounds_or_events", 0))
    if summary.get("method") == "sync_fedavg":
        return progress * float(config.get("clients", 10))
    return progress


def signature_matches(config: dict) -> bool:
    if not signature:
        return True
    for item in signature.split(","):
        if not item:
            continue
        key, expected = item.split("=", 1)
        actual = config.get(key)
        try:
            if abs(float(actual) - float(expected)) > 1e-9:
                return False
        except (TypeError, ValueError):
            if str(actual) != expected:
                return False
    return True


for path in glob.glob("results/*_summary.json"):
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    config = summary.get("config", {})
    if summary.get("method") != method:
        continue
    if dataset_of(path, summary) != dataset:
        continue
    if int(config.get("seed", 42)) != seed:
        continue
    if str(config.get("model", "qwen")) != model:
        continue
    if abs(update_budget(summary, config) - budget) > 1e-6:
        continue
    if not signature_matches(config):
        continue
    sys.exit(0)
sys.exit(1)
PY
}

latest_eval_line() {
  if [[ -n "${current_log:-}" && -f "$current_log" ]]; then
    grep -E 'event=|round=' "$current_log" | tail -n 1 || true
  fi
}

metric_snapshot() {
  python - <<'PY'
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


def dataset_of(path: Path, summary: dict) -> str:
    config = summary.get("config", {})
    if config.get("dataset"):
        return str(config["dataset"])
    method = summary.get("method", "")
    prefix = f"{method}_"
    if path.name.startswith(prefix):
        parts = path.name[len(prefix):].split("_")
        if len(parts) >= 4:
            return "_".join(parts[:-3])
    return "pathmnist"


rows = []
for raw in glob.glob("results/*_summary.json"):
    path = Path(raw)
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    rows.append(
        {
            "dataset": dataset_of(path, summary),
            "method": summary.get("method", ""),
            "seed": int(config.get("seed", 42)),
            "model": str(config.get("model", "qwen")),
            "partition": str(config.get("partition", "")),
            "best": float(summary.get("best_test_acc", -1.0)),
            "final": float(summary.get("final_test_acc", -1.0)),
        }
    )

print(f"summary_count={len(rows)}")
if rows:
    latest = rows[-1]
    print(
        "latest_summary="
        f"{latest['dataset']}/{latest['method']}/seed{latest['seed']}/{latest['model']} "
        f"best={latest['best']:.4f} final={latest['final']:.4f}"
    )
    caa = [row for row in rows if row["method"] == "caa_fedbuff_v2"]
    baselines = [row for row in rows if row["method"] in {"sync_fedavg", "naive_async", "staleness_async", "fedbuff_async"}]
    if caa and baselines:
        best_caa = max(caa, key=lambda row: row["best"])
        best_base = max(baselines, key=lambda row: row["best"])
        print(
            "best_caa_v2_gap_to_strongest_baseline="
            f"{best_base['best'] - best_caa['best']:.4f} "
            f"caa={best_caa['dataset']}/{best_caa['best']:.4f} "
            f"base={best_base['method']}/{best_base['dataset']}/{best_base['best']:.4f}"
        )
PY
}

run_stage0() {
  log_action "stage0 py_compile"
  PYTHONPYCACHEPREFIX=/tmp/fedpath_distsys24_pycache python -m py_compile src/fed_pathmnist/*.py scripts/*.py >> "$ACTION_LOG" 2>&1 || log_action "stage0 py_compile failed"
  log_action "stage0 synthetic caa_fedbuff_v2 smoke"
  PYTHONPATH=src python src/fed_pathmnist/run.py \
    --synthetic --method caa_fedbuff_v2 --events 8 --clients 2 \
    --buffer-size 2 --model small_cnn --device cpu \
    --result-dir /tmp/fedpath_distsys24_smoke_results \
    --checkpoint-dir /tmp/fedpath_distsys24_smoke_checkpoints >> "$ACTION_LOG" 2>&1 || log_action "stage0 synthetic smoke failed"
  rm -rf /tmp/fedpath_distsys24_smoke_results /tmp/fedpath_distsys24_smoke_checkpoints /tmp/fedpath_distsys24_pycache
}

start_action() {
  local action="$1"
  IFS='|' read -r stage dataset method seed model partition dirichlet_alpha delay_key rounds events budget label extra signature variant <<< "$action"
  # shellcheck disable=SC2206
  local extra_array=($extra)
  # shellcheck disable=SC2206
  local delay_array=($(delay_args_for_key "$delay_key"))

  if summary_exists "$dataset" "$method" "$seed" "$model" "$budget" "$signature"; then
    last_decision="skip existing ${label}"
    log_action "$last_decision"
    return 1
  fi
  if [[ -n "$(active_training_commands)" ]]; then
    last_decision="wanted to start ${label}, but a training process is already active"
    log_action "$last_decision"
    return 1
  fi

  current_stage="$stage"
  current_action="$label"
  current_log="$LOG_DIR/${STAMP}_${label}.log"
  local cmd=(
    PYTHONPATH=src python src/fed_pathmnist/run.py
    --dataset "$dataset"
    --method "$method"
    --seed "$seed"
    --model "$model"
    --partition "$partition"
    "${COMMON_ARGS[@]}"
    "${extra_array[@]}"
  )

  if [[ "$partition" == "dirichlet" ]]; then
    cmd+=(--dirichlet-alpha "$dirichlet_alpha")
  fi

  if [[ "$method" == "sync_fedavg" ]]; then
    cmd+=(--rounds "$rounds")
  else
    cmd+=(--events "$events" "${ASYNC_ARGS[@]}" "${delay_array[@]}")
    case "$method" in
      naive_async)
        cmd+=(--alpha 0.5)
        ;;
      staleness_async)
        cmd+=(--alpha 0.5 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.1)
        ;;
      fedbuff_async)
        cmd+=(--buffer-size 5 --alpha 0.5 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.05)
        ;;
      agreement_fedbuff_async)
        cmd+=("${CAA_ARGS[@]}")
        ;;
      caa_fedbuff_v2)
        cmd+=("${CAA_ARGS[@]}" --server-delta-momentum 0.8)
        case "$variant" in
          no_server_ema)
            cmd+=(--history-agreement-blend 0.0 --client-fairness-power 0.5)
            ;;
          no_fairness)
            cmd+=(--history-agreement-blend 0.25 --client-fairness-power 0.0)
            ;;
          no_clipping)
            cmd+=(--history-agreement-blend 0.25 --client-fairness-power 0.5 --delta-clip-multiplier 1000000.0)
            ;;
          static_alpha)
            cmd+=(--history-agreement-blend 0.25 --client-fairness-power 0.5 --adaptive-alpha-min 0.62 --adaptive-alpha-max 0.62)
            ;;
          *)
            cmd+=(--history-agreement-blend 0.25 --client-fairness-power 0.5)
            ;;
        esac
        ;;
    esac
  fi

  log_action "START ${label}"
  log_action "CMD ${cmd[*]}"
  env "${cmd[@]}" > "$current_log" 2>&1 &
  current_pid="$!"
  last_decision="started ${label}; monitor pid=${current_pid}"
  return 0
}

heartbeat() {
  local cycle="$1"
  {
    echo "===== heartbeat ${cycle}/${CYCLES} $(date '+%F %T') ====="
    echo "marker=$MARKER"
    echo "stage=${current_stage:-idle}"
    echo "queue_index=${action_index}/${#actions[@]}"
    echo "current_action=${current_action:-}"
    echo "current_pid=${current_pid:-}"
    echo "failed_runs=${#failed_runs[@]}"
    if [[ "${#failed_runs[@]}" -gt 0 ]]; then
      printf 'failed_labels=%s\n' "${failed_runs[*]}"
    fi
    echo "active_training:"
    active_training_commands
    echo "gpu:"
    gpu_status
    echo "counts:"
    echo "csv=$(find results -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ') summary=$(find results -maxdepth 1 -name '*_summary.json' | wc -l | tr -d ' ') checkpoints=$(find checkpoints -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    echo "metrics:"
    metric_snapshot
    echo "latest_eval:"
    latest_eval_line
    echo "next_action=${last_decision:-}"
    echo
  } >> "$HEARTBEAT_LOG"
}

build_queue
log_action "queue_size=${#actions[@]} cycles=${CYCLES} interval=${INTERVAL_SECONDS} poll=${POLL_SECONDS}"

current_pid=""
current_stage=""
current_action=""
current_log=""
last_decision="stage0 sanity and report refresh"
action_index=0

heartbeat_count=0
next_heartbeat=0
next_idle_report=0

while [[ "$heartbeat_count" -lt "$CYCLES" ]]; do
  if [[ -n "$current_pid" ]] && ! kill -0 "$current_pid" 2>/dev/null; then
    wait "$current_pid"
    status="$?"
    log_action "DONE ${current_action} status=${status}"
    if [[ "$status" -ne 0 ]]; then
      failed_runs+=("${current_action}")
    fi
    current_pid=""
    current_stage=""
    current_action=""
    current_log=""
  fi

  while [[ -z "$current_pid" && "$action_index" -lt "${#actions[@]}" ]]; do
    action="${actions[$action_index]}"
    action_index=$((action_index + 1))
    start_action "$action" || true
  done

  now="$(date +%s)"
  if [[ -z "$current_pid" && "$action_index" -ge "${#actions[@]}" ]]; then
    current_stage="idle_qa"
    last_decision="queue complete; idle QA heartbeat continues until ${CYCLES}/${CYCLES}"
    if [[ "$next_idle_report" -eq 0 || "$now" -ge "$next_idle_report" ]]; then
      next_idle_report=$((now + IDLE_REPORT_SECONDS))
    fi
    break
  fi
  

  if [[ "$next_heartbeat" -eq 0 || "$now" -ge "$next_heartbeat" ]]; then
    heartbeat_count=$((heartbeat_count + 1))
    heartbeat "$heartbeat_count"
    next_heartbeat=$((now + INTERVAL_SECONDS))
  fi

  sleep "$POLL_SECONDS"
done

if [[ -n "$current_pid" ]]; then
  log_action "heartbeat cycles completed; waiting for active run ${current_action} pid=${current_pid}"
  wait "$current_pid"
  status="$?"
  log_action "DONE ${current_action} status=${status}"
  if [[ "$status" -ne 0 ]]; then
    failed_runs+=("${current_action}")
  fi
fi

log_action "distributed systems 24h runner finished failed_runs=${#failed_runs[@]}"
