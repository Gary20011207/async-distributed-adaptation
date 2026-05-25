#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

ACTION_LOG="$LOG_DIR/missing_seed_completion_${STAMP}.log"
HEARTBEAT_LOG="$LOG_DIR/missing_seed_completion_heartbeat_${STAMP}.log"

COMMON_ARGS=(
  --clients 10
  --batch-size 128
  --lr 0.01
  --lr-scheduler cosine
  --min-lr 0.0001
  --local-epochs 1
  --augment
  --device cuda
  --save-best
)

ASYNC_ARGS=(
  --eval-every 20
  --delay-mode heterogeneous
  --straggler-ratio 0.2
  --straggler-multiplier 5.0
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

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ACTION_LOG"
}

dataset_extra_args() {
  local dataset="$1"
  case "$dataset" in
    organamnist|organcmnist|organsmnist)
      echo "--max-train-samples 15000 --max-test-samples 4000"
      ;;
    octmnist)
      echo "--max-train-samples 15000 --max-test-samples 3000"
      ;;
    tissuemnist)
      echo "--max-train-samples 20000 --max-test-samples 5000"
      ;;
    *)
      echo ""
      ;;
  esac
}

summary_exists() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local budget="$4"
  python - "$dataset" "$method" "$seed" "$budget" <<'PY'
from __future__ import annotations

import glob
import json
import sys

dataset, method, seed, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])

for path in glob.glob("results/*_summary.json"):
    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        continue
    config = summary.get("config", {}) or {}
    if summary.get("method") != method:
        continue
    if config.get("dataset", "pathmnist") != dataset:
        continue
    if config.get("model", "resnet18") != "resnet18":
        continue
    if config.get("partition", "iid") != "iid":
        continue
    if int(config.get("seed", 42)) != seed:
        continue
    if config.get("synthetic"):
        continue
    if config.get("update_budget") not in (None, ""):
        observed_budget = float(config["update_budget"])
    elif method == "sync_fedavg":
        observed_budget = float(summary.get("total_rounds_or_events", 0)) * float(config.get("clients", 10))
    else:
        observed_budget = float(summary.get("total_rounds_or_events", 0))
    if abs(observed_budget - budget) < 1e-6:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

heartbeat() {
  local completed="$1"
  local total="$2"
  {
    echo "timestamp=$(date '+%F %T')"
    echo "progress=${completed}/${total}"
    echo "active_training=$(pgrep -af 'src/fed_pathmnist/run.py' || true)"
    echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"
    echo "summary_count=$(find results -maxdepth 1 -name '*_summary.json' | wc -l)"
    echo "---"
  } >> "$HEARTBEAT_LOG"
}

run_one() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local rounds=30
  local events=300
  local budget=300
  local extra
  local cmd
  extra="$(dataset_extra_args "$dataset")"

  if summary_exists "$dataset" "$method" "$seed" "$budget"; then
    log "SKIP existing dataset=${dataset} method=${method} seed=${seed} budget=${budget}"
    return 0
  fi

  if [[ "$method" == "sync_fedavg" ]]; then
    cmd=(python src/fed_pathmnist/run.py --dataset "$dataset" --method "$method" --rounds "$rounds" --seed "$seed" "${COMMON_ARGS[@]}")
  else
    cmd=(python src/fed_pathmnist/run.py --dataset "$dataset" --method "$method" --events "$events" --seed "$seed" "${COMMON_ARGS[@]}" "${ASYNC_ARGS[@]}")
    if [[ "$method" == "caa_fedbuff_v2" ]]; then
      cmd+=("${CAA_ARGS[@]}")
    elif [[ "$method" == "staleness_async" ]]; then
      cmd+=(--alpha 0.5 --staleness-decay inverse)
    elif [[ "$method" == "naive_async" ]]; then
      cmd+=(--alpha 0.5)
    fi
  fi

  # shellcheck disable=SC2206
  local extra_array=($extra)
  cmd+=("${extra_array[@]}")

  local run_log="$LOG_DIR/missing_${dataset}_${method}_seed${seed}_${STAMP}.log"
  log "RUN dataset=${dataset} method=${method} seed=${seed}"
  log "CMD PYTHONPATH=src ${cmd[*]}"
  if PYTHONPATH=src "${cmd[@]}" > "$run_log" 2>&1; then
    log "DONE dataset=${dataset} method=${method} seed=${seed} log=${run_log}"
  else
    log "FAIL dataset=${dataset} method=${method} seed=${seed} log=${run_log}"
    return 1
  fi
}

refresh_report() {
  log "Refreshing report summaries and figures"
  PYTHONPATH=src python -m fed_pathmnist.plot_results --csv results/*.csv --outdir figures >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_report_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_distributed_systems_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_classification_results.py --result-dir results --checkpoint-dir checkpoints --outdir figures/classification >> "$ACTION_LOG" 2>&1 || true
  python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md >> "$ACTION_LOG" 2>&1 || true
}

main() {
  local queue=()
  local dataset method seed
  local methods=(sync_fedavg naive_async staleness_async caa_fedbuff_v2)

  for dataset in dermamnist octmnist breastmnist tissuemnist organcmnist; do
    for method in "${methods[@]}"; do
      for seed in 42 43 44; do
        queue+=("${dataset}|${method}|${seed}")
      done
    done
  done

  local total="${#queue[@]}"
  local completed=0
  local failures=0
  log "Starting missing seed completion total=${total}"
  heartbeat "$completed" "$total"

  for item in "${queue[@]}"; do
    IFS='|' read -r dataset method seed <<< "$item"
    if run_one "$dataset" "$method" "$seed"; then
      :
    else
      failures=$((failures + 1))
    fi
    completed=$((completed + 1))
    heartbeat "$completed" "$total"
  done

  refresh_report
  log "Completed missing seed completion failures=${failures}"
  heartbeat "$completed" "$total"
  return "$failures"
}

main "$@"
