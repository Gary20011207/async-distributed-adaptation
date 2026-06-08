#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-fedpath-r139}" || true
fi
cd "$ROOT_DIR"
mkdir -p logs figures/report presentation
STAMP="$(date +%Y%m%d_%H%M%S)"
HEARTBEAT_LOG="logs/report_polish3h_heartbeat_${STAMP}.log"
ACTION_LOG="logs/report_polish3h_actions_${STAMP}.log"
FAIL_LOG="logs/report_polish3h_failures_${STAMP}.log"
: > "$HEARTBEAT_LOG"
: > "$ACTION_LOG"
: > "$FAIL_LOG"

CYCLES=18
INTERVAL=600
POLL=30

log_action() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$ACTION_LOG"
}

run_step() {
  local tag="$1"
  shift
  local log="logs/${tag}_${STAMP}.log"
  log_action "START $tag"
  "$@" > "$log" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    echo "$(date --iso-8601=seconds) $tag status=$status log=$log" >> "$FAIL_LOG"
    log_action "FAIL $tag status=$status"
  else
    log_action "DONE $tag"
  fi
  return $status
}

refresh_report() {
  run_step py_compile python -m py_compile src/fed_mllm/*.py scripts/*.py || true
  run_step plot_results bash -lc "PYTHONPATH=src python -m fed_mllm.plot_results --csv 'results/*.csv' --outdir figures" || true
  run_step plot_report_summary python scripts/plot_report_summary.py --result-dir results --outdir figures/report || true
  run_step summarize_results python scripts/summarize_results.py --result-dir results --out REPORT_NOTES.md || true
  run_step polish_report python scripts/polish_mllm_report.py --result-dir results --outdir figures/report --presentation-dir presentation || true
}

heartbeat() {
  local cycle="$1"
  local gpu="nvidia-smi unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true)"
  fi
  local summary_count="$(find results -name '*_summary.json' 2>/dev/null | wc -l)"
  local csv_count="$(find results -name '*.csv' 2>/dev/null | wc -l)"
  local real_rows="0"
  if [[ -f figures/report/mean_std_summary.csv ]]; then
    real_rows="$(python - <<'PY2' 2>/dev/null || echo 0
import pandas as pd
p='figures/report/mean_std_summary.csv'
df=pd.read_csv(p)
print(int((df.get('backend_group','') == 'real_qwenvl').sum()) if 'backend_group' in df else 0)
PY2
)"
  fi
  {
    echo "===== heartbeat ${cycle}/${CYCLES} $(date --iso-8601=seconds) ====="
    echo "gpu=${gpu}"
    echo "summary_count=${summary_count} csv_count=${csv_count}"
    echo "real_qwenvl_meanstd_rows=${real_rows}"
    echo "failures=$(wc -l < "$FAIL_LOG")"
    echo "report_notes=$([[ -f REPORT_NOTES.md ]] && wc -l < REPORT_NOTES.md || echo 0) lines"
    echo "presentation_md=$([[ -f presentation/mllm_report_zh.md ]] && wc -l < presentation/mllm_report_zh.md || echo 0) lines"
    echo "diagnostics=$([[ -f figures/report/qwenvl_closed_answer_diagnostics.csv ]] && wc -l < figures/report/qwenvl_closed_answer_diagnostics.csv || echo 0) lines"
  } >> "$HEARTBEAT_LOG"
}

log_action "report polish runner started"
refresh_report
cycle=1
last_refresh=$(date +%s)
last_heartbeat=0
while [[ $cycle -le $CYCLES ]]; do
  now=$(date +%s)
  if [[ $last_heartbeat -eq 0 || $((now - last_heartbeat)) -ge $INTERVAL ]]; then
    heartbeat "$cycle"
    cycle=$((cycle + 1))
    last_heartbeat=$now
  fi
  if [[ $((now - last_refresh)) -ge 3600 ]]; then
    log_action "hourly refresh"
    refresh_report
    last_refresh=$now
  fi
  sleep "$POLL"
done
log_action "report polish runner completed cycles=$CYCLES"
refresh_report
