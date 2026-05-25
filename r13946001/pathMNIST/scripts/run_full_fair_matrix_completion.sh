#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

ACTION_LOG="$LOG_DIR/full_fair_matrix_completion_${STAMP}.log"
HEARTBEAT_LOG="$LOG_DIR/full_fair_matrix_completion_heartbeat_${STAMP}.log"

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

BUFF_ARGS=(
  --buffer-size 5
  --alpha 0.5
  --staleness-decay inverse
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
  --server-delta-momentum 0.8
  --history-agreement-blend 0.25
  --client-fairness-power 0.5
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

rounds_for_dataset() {
  local dataset="$1"
  if [[ "$dataset" == "pathmnist" ]]; then
    echo 100
  else
    echo 30
  fi
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
import math
import sys

dataset, method, seed, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])


def as_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def eq_float(actual, expected, tol=1e-9):
    actual = as_float(actual)
    return actual is not None and abs(actual - expected) <= tol


def eq_int(actual, expected):
    actual = as_float(actual)
    return actual is not None and int(actual) == expected


def eq_bool(actual, expected):
    return bool(actual) is expected


def eq_str(actual, expected):
    if actual in (None, ""):
        actual = "resnet18" if expected == "resnet18" else actual
    return str(actual) == expected


def expected_samples(ds):
    if ds in {"organamnist", "organcmnist", "organsmnist"}:
        return 15000, 4000
    if ds == "octmnist":
        return 15000, 3000
    if ds == "tissuemnist":
        return 20000, 5000
    return None, None


def observed_budget(summary, config):
    if config.get("update_budget") not in (None, ""):
        return float(config["update_budget"])
    if method == "sync_fedavg":
        return float(summary.get("total_rounds_or_events", 0)) * float(config.get("clients", 10))
    return float(summary.get("total_rounds_or_events", 0))


def common_ok(config):
    train_limit, test_limit = expected_samples(dataset)
    checks = [
        eq_str(config.get("model", "resnet18"), "resnet18"),
        str(config.get("partition", "iid") or "iid") == "iid",
        eq_int(config.get("clients", 10), 10),
        eq_int(config.get("batch_size", 128), 128),
        eq_float(config.get("lr", 0.01), 0.01),
        str(config.get("lr_scheduler", "cosine")) == "cosine",
        eq_float(config.get("min_lr", 0.0001), 0.0001),
        eq_int(config.get("local_epochs", 1), 1),
        eq_bool(config.get("augment", True), True),
        config.get("max_train_samples") == train_limit,
        config.get("max_test_samples") == test_limit,
    ]
    return all(checks)


def async_ok(config):
    return (
        str(config.get("delay_mode")) == "heterogeneous"
        and eq_float(config.get("straggler_ratio"), 0.2)
        and eq_float(config.get("straggler_multiplier"), 5.0)
        and eq_int(config.get("eval_every", 20), 20)
    )


def method_ok(config):
    if method == "sync_fedavg":
        return True
    if not async_ok(config):
        return False
    if method == "naive_async":
        return eq_float(config.get("alpha"), 0.5)
    if method == "staleness_async":
        return eq_float(config.get("alpha"), 0.5) and str(config.get("staleness_decay")) == "inverse"
    if method == "fedbuff_async":
        return (
            eq_float(config.get("alpha"), 0.5)
            and eq_int(config.get("buffer_size"), 5)
            and str(config.get("staleness_decay")) == "inverse"
        )
    if method in {"agreement_fedbuff_async", "caa_fedbuff_v2"}:
        checks = [
            eq_int(config.get("buffer_size"), 5),
            eq_float(config.get("alpha"), 0.62),
            str(config.get("staleness_decay")) == "hinge",
            eq_float(config.get("staleness_hinge_b"), 5.0),
            eq_float(config.get("staleness_hinge_a"), 0.05),
            eq_float(config.get("agreement_epsilon"), 0.15),
            eq_float(config.get("agreement_power"), 0.5),
            eq_float(config.get("agreement_drop_threshold"), -0.05),
            eq_float(config.get("delta_clip_multiplier"), 1.8),
            eq_float(config.get("adaptive_alpha_min"), 0.20),
            eq_float(config.get("adaptive_alpha_max"), 0.70),
            eq_float(config.get("adaptive_alpha_boost"), 0.25),
            eq_float(config.get("adaptive_staleness_scale"), 10.0),
        ]
        if method == "caa_fedbuff_v2":
            checks.extend([
                eq_float(config.get("server_delta_momentum"), 0.8),
                eq_float(config.get("history_agreement_blend"), 0.25),
                eq_float(config.get("client_fairness_power"), 0.5),
            ])
        return all(checks)
    return False


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
    if int(config.get("seed", 42)) != seed:
        continue
    if config.get("synthetic"):
        continue
    if abs(observed_budget(summary, config) - budget) > 1e-6:
        continue
    if common_ok(config) and method_ok(config):
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
  local rounds
  local events
  local budget
  local extra
  local cmd

  rounds="$(rounds_for_dataset "$dataset")"
  events=$((rounds * 10))
  budget="$events"
  extra="$(dataset_extra_args "$dataset")"

  if summary_exists "$dataset" "$method" "$seed" "$budget"; then
    log "SKIP existing dataset=${dataset} method=${method} seed=${seed} budget=${budget}"
    return 0
  fi

  if [[ "$method" == "sync_fedavg" ]]; then
    cmd=(python src/fed_pathmnist/run.py --dataset "$dataset" --method "$method" --rounds "$rounds" --seed "$seed" "${COMMON_ARGS[@]}")
  else
    cmd=(python src/fed_pathmnist/run.py --dataset "$dataset" --method "$method" --events "$events" --seed "$seed" "${COMMON_ARGS[@]}" "${ASYNC_ARGS[@]}")
    case "$method" in
      fedbuff_async)
        cmd+=("${BUFF_ARGS[@]}")
        ;;
      agreement_fedbuff_async|caa_fedbuff_v2)
        cmd+=("${CAA_ARGS[@]}")
        ;;
      staleness_async)
        cmd+=(--alpha 0.5 --staleness-decay inverse)
        ;;
      naive_async)
        cmd+=(--alpha 0.5)
        ;;
    esac
  fi

  # shellcheck disable=SC2206
  local extra_array=($extra)
  cmd+=("${extra_array[@]}")

  local run_log="$LOG_DIR/fullmatrix_${dataset}_${method}_seed${seed}_${STAMP}.log"
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
  local datasets=(pathmnist pneumoniamnist bloodmnist organamnist dermamnist octmnist breastmnist tissuemnist organcmnist)
  local methods=(sync_fedavg naive_async staleness_async fedbuff_async agreement_fedbuff_async caa_fedbuff_v2)
  local seeds=(42 43 44)
  local queue=()
  local dataset method seed item

  for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
      for seed in "${seeds[@]}"; do
        queue+=("${dataset}|${method}|${seed}")
      done
    done
  done

  local total="${#queue[@]}"
  local completed=0
  local failures=0
  log "Starting full fair matrix completion total=${total}"
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
  log "Completed full fair matrix completion failures=${failures}"
  heartbeat "$completed" "$total"
  return "$failures"
}

main "$@"
