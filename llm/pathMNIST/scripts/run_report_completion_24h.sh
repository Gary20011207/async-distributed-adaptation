#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

HEARTBEAT_LOG="$LOG_DIR/report24_heartbeat_${STAMP}.log"
ACTION_LOG="$LOG_DIR/report24_actions_${STAMP}.log"
MARKER="$LOG_DIR/report24_${STAMP}.marker"
touch "$MARKER"

CYCLES="${CYCLES:-144}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
POLL_SECONDS="${POLL_SECONDS:-30}"

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
  --delay-mode heterogeneous
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

log_action() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$ACTION_LOG"
}

add_action() {
  actions+=("$*")
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

enqueue_run() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local model="$4"
  local label="$5"
  local signature="${6:-}"
  local rounds
  local events
  local budget
  local extra
  rounds="$(rounds_for_dataset "$dataset")"
  events=$((rounds * 10))
  budget="$events"
  extra="$(dataset_extra_args "$dataset")"
  add_action "$dataset|$method|$seed|$model|$rounds|$events|$budget|$label|$extra|$signature"
}

build_queue() {
  local methods=(sync_fedavg naive_async staleness_async agreement_fedbuff_async caa_fedbuff_v2)
  local dataset
  local method

  # Required extension: additional MedMNIST datasets with fair single-seed coverage.
  for dataset in breastmnist tissuemnist; do
    for method in "${methods[@]}"; do
      enqueue_run "$dataset" "$method" 42 resnet18 "coverage_${dataset}_${method}"
    done
  done

  # Required backbone sanity check: the algorithm should not be ResNet-only.
  for dataset in pneumoniamnist bloodmnist organamnist; do
    for method in sync_fedavg naive_async staleness_async caa_fedbuff_v2; do
      enqueue_run "$dataset" "$method" 42 small_cnn "backbone_smallcnn_${dataset}_${method}"
    done
  done

  # Mechanism ablation: keep the CAA-v2 rule simple and explainable.
  enqueue_run pathmnist caa_fedbuff_v2 42 resnet18 "ablation_path_full_caa_v2" "server_delta_momentum=0.8,history_agreement_blend=0.25,client_fairness_power=0.5"
  enqueue_run pathmnist caa_fedbuff_v2 42 resnet18 "ablation_path_no_ema" "server_delta_momentum=0.8,history_agreement_blend=0.0,client_fairness_power=0.5"
  enqueue_run pathmnist caa_fedbuff_v2 42 resnet18 "ablation_path_no_fairness" "server_delta_momentum=0.8,history_agreement_blend=0.25,client_fairness_power=0.0"
  enqueue_run pathmnist agreement_fedbuff_async 42 resnet18 "ablation_path_old_caa"

  enqueue_run bloodmnist caa_fedbuff_v2 42 resnet18 "ablation_blood_full_caa_v2" "server_delta_momentum=0.8,history_agreement_blend=0.25,client_fairness_power=0.5"
  enqueue_run bloodmnist caa_fedbuff_v2 42 resnet18 "ablation_blood_no_ema" "server_delta_momentum=0.8,history_agreement_blend=0.0,client_fairness_power=0.5"
  enqueue_run bloodmnist caa_fedbuff_v2 42 resnet18 "ablation_blood_no_fairness" "server_delta_momentum=0.8,history_agreement_blend=0.25,client_fairness_power=0.0"
  enqueue_run bloodmnist agreement_fedbuff_async 42 resnet18 "ablation_blood_old_caa"

  # Optional coverage if the required queue finishes early.
  for method in sync_fedavg naive_async staleness_async caa_fedbuff_v2; do
    enqueue_run organcmnist "$method" 42 resnet18 "optional_organc_${method}"
  done
  for method in sync_fedavg naive_async staleness_async caa_fedbuff_v2; do
    enqueue_run breastmnist "$method" 42 mobilenet_v3_small "optional_mobilenet_breast_${method}"
  done
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
    if str(config.get("model", "resnet18")) != model:
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
from collections import defaultdict
from pathlib import Path

required_primary = ["pathmnist", "pneumoniamnist", "bloodmnist", "organamnist"]
required_methods = ["sync_fedavg", "naive_async", "staleness_async", "agreement_fedbuff_async", "caa_fedbuff_v2"]
required_seeds = [42, 43, 44]
new_datasets = ["breastmnist", "tissuemnist"]


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


def budget(summary: dict, config: dict) -> float:
    if config.get("update_budget") not in (None, ""):
        return float(config["update_budget"])
    progress = float(summary.get("total_rounds_or_events", 0))
    if summary.get("method") == "sync_fedavg":
        return progress * float(config.get("clients", 10))
    return progress


have = set()
small_cnn = set()
new_coverage = defaultdict(set)
best_caa_family = None
latest = None
rows = []
for path in Path("results").glob("*_summary.json"):
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    dataset = dataset_of(path, summary)
    method = summary.get("method", "")
    seed = int(config.get("seed", 42))
    model = str(config.get("model", "resnet18"))
    acc = float(summary.get("best_test_acc", -1.0))
    final = float(summary.get("final_test_acc", -1.0))
    rows.append((dataset, method, seed, model, acc, final))
    if dataset in required_primary and method in required_methods and seed in required_seeds and model == "resnet18":
        expected = 1000 if dataset == "pathmnist" else 300
        if abs(budget(summary, config) - expected) < 1e-6:
            have.add((dataset, method, seed))
    if dataset in new_datasets and method in required_methods and seed == 42 and model == "resnet18":
        new_coverage[dataset].add(method)
    if dataset in {"pneumoniamnist", "bloodmnist", "organamnist"} and model == "small_cnn":
        small_cnn.add((dataset, method))
    if method in {"agreement_fedbuff_async", "caa_fedbuff_v2"}:
        if best_caa_family is None or acc > best_caa_family[0]:
            best_caa_family = (acc, dataset, method, seed, final)
    latest = (dataset, method, seed, model, acc, final)

complete_primary = 0
missing_primary = []
for dataset in required_primary:
    for seed in required_seeds:
        if all((dataset, method, seed) in have for method in required_methods):
            complete_primary += 1
        else:
            missing_primary.append(f"{dataset}:seed{seed}")

print(f"summary_count={len(rows)}")
print(f"primary_seed_sets={complete_primary}/12 missing={len(set(missing_primary))}")
print(
    "new_dataset_coverage="
    + ",".join(f"{dataset}:{len(new_coverage[dataset])}/5" for dataset in new_datasets)
)
print(f"small_cnn_method_rows={len(small_cnn)}/12")
if best_caa_family:
    acc, dataset, method, seed, final = best_caa_family
    print(f"best_caa_family={acc:.4f} dataset={dataset} method={method} seed={seed} final={final:.4f}")
if latest:
    dataset, method, seed, model, acc, final = latest
    print(f"latest_summary={dataset}/{method}/seed{seed}/{model} best={acc:.4f} final={final:.4f}")
PY
}

generate_report_pack() {
  log_action "regenerate report pack"
  PYTHONPATH=src python -m fed_pathmnist.plot_results --csv results/*.csv --outdir figures >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_report_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_seeded_summary.py --result-dir results --outdir figures/report >> "$ACTION_LOG" 2>&1 || true
  python scripts/plot_classification_results.py --result-dir results --checkpoint-dir checkpoints --outdir figures/classification --datasets pathmnist pneumoniamnist bloodmnist organamnist breastmnist tissuemnist >> "$ACTION_LOG" 2>&1 || true
  python scripts/summarize_results.py --result-dir results --out ../REPORT_NOTES.md >> "$ACTION_LOG" 2>&1 || true
}

start_action() {
  local action="$1"
  IFS='|' read -r dataset method seed model rounds events budget label extra signature <<< "$action"
  # shellcheck disable=SC2206
  local extra_array=($extra)

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

  current_action="$label"
  current_log="$LOG_DIR/${STAMP}_${label}.log"
  local cmd=(
    PYTHONPATH=src python src/fed_pathmnist/run.py
    --dataset "$dataset"
    --method "$method"
    --seed "$seed"
    --model "$model"
    "${COMMON_ARGS[@]}"
    "${extra_array[@]}"
  )

  if [[ "$method" == "sync_fedavg" ]]; then
    cmd+=(--rounds "$rounds")
  else
    cmd+=(--events "$events" "${ASYNC_ARGS[@]}")
    case "$method" in
      naive_async)
        cmd+=(--alpha 0.5)
        ;;
      staleness_async)
        cmd+=(--alpha 0.5 --staleness-decay hinge --staleness-hinge-b 5 --staleness-hinge-a 0.1)
        ;;
      agreement_fedbuff_async)
        cmd+=("${CAA_ARGS[@]}")
        ;;
      caa_fedbuff_v2)
        cmd+=("${CAA_ARGS[@]}" --server-delta-momentum 0.8)
        if [[ "$label" == *"_no_ema" ]]; then
          cmd+=(--history-agreement-blend 0.0)
        else
          cmd+=(--history-agreement-blend 0.25)
        fi
        if [[ "$label" == *"_no_fairness" ]]; then
          cmd+=(--client-fairness-power 0.0)
        else
          cmd+=(--client-fairness-power 0.5)
        fi
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
    echo "queue_index=${action_index}/${#actions[@]}"
    echo "current_action=${current_action:-}"
    echo "current_pid=${current_pid:-}"
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
    echo "next_decision=${last_decision:-}"
    echo
  } >> "$HEARTBEAT_LOG"
}

build_queue
log_action "queue_size=${#actions[@]} cycles=${CYCLES} interval=${INTERVAL_SECONDS} poll=${POLL_SECONDS}"

current_pid=""
current_action=""
current_log=""
last_decision="refresh existing report pack before queue"
action_index=0

generate_report_pack

heartbeat_count=0
next_heartbeat=0

while [[ "$heartbeat_count" -lt "$CYCLES" ]]; do
  if [[ -n "$current_pid" ]] && ! kill -0 "$current_pid" 2>/dev/null; then
    wait "$current_pid"
    status="$?"
    log_action "DONE ${current_action} status=${status}"
    current_pid=""
    current_action=""
    current_log=""
    generate_report_pack
  fi

  while [[ -z "$current_pid" && "$action_index" -lt "${#actions[@]}" ]]; do
    action="${actions[$action_index]}"
    action_index=$((action_index + 1))
    start_action "$action" || true
  done

  if [[ -z "$current_pid" && "$action_index" -ge "${#actions[@]}" ]]; then
    last_decision="queue complete; final report pack generated"
    generate_report_pack
    heartbeat_count=$((heartbeat_count + 1))
    heartbeat "$heartbeat_count"
    log_action "report completion queue finished early at heartbeat=${heartbeat_count}"
    exit 0
  fi

  now="$(date +%s)"
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
fi

generate_report_pack
log_action "24h report completion runner finished"
