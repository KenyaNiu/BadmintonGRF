#!/usr/bin/env bash
# =============================================================================
# BadmintonGRF — 顺序跑注册基线（train → fuse → 可选 aggregate）
#
# 用法：
#   export BADMINTON_DATA_ROOT=/path/to/data
#   bash run_all_baselines.sh
#   bash run_all_baselines.sh train tcn_bilstm
#
# 若训练出现 DataLoader worker SIGSEGV（OpenMP/fork）：
#   export NUM_WORKERS=0
# 可选：export NO_PIN_MEMORY=1  （脚本会传 --no_pin_memory）
# 更大 batch 吃满显存、加速（按卡调整）：export BATCH_SIZE=768
#
# 推荐主入口：python -m baseline <train|fuse|aggregate|...>
# 长时间全量跑批（重试 + fold 续跑 + 结束后汇总）：run_benchmark_resilient.sh
# bug 修完再续跑（等人 touch 继续）：run_benchmark_wait_fix_loop.sh + BENCHMARK_RUN_ROOT
# fold 续跑：python -m baseline train 默认开启；关闭：export BADMINTON_RESUME_FOLDS=0 或传 --no_resume_folds
# =============================================================================
# bash 才有 pipefail；用 sh/dash 跑会报 "set: pipefail: invalid option name"
set -eu
if [ -n "${BASH_VERSION:-}" ]; then
  set -o pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${BADMINTON_DATA_ROOT:-$SCRIPT_DIR/data}"
LOSO="${LOSO_SPLITS:-$DATA_ROOT/reports/loso_splits_10p.json}"
MODE="${1:-all}"
METHOD_ARG="${2:-}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${SCRIPT_DIR}/runs/benchmark_runs_${TS}"
LOG_DIR="${SCRIPT_DIR}/runs/logs_${TS}"
mkdir -p "$RUN_ROOT" "$LOG_DIR"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

METHODS_FLAT=(tcn_bilstm gru_bigru tcn_mlp seq_transformer dlinear patch_tst ms_tcn patch_tst_xl tsmixer_grf)
METHODS_STGCN=(stgcn_transformer)

# Optional: NUM_WORKERS=0, NO_PIN_MEMORY=1 见文件头注释
TRAIN_EXTRA=()
if [ -n "${NUM_WORKERS+set}" ]; then
  TRAIN_EXTRA+=(--num_workers "${NUM_WORKERS}")
fi
if [ "${NO_PIN_MEMORY:-0}" = "1" ]; then
  TRAIN_EXTRA+=(--no_pin_memory)
fi
if [ -n "${BATCH_SIZE+set}" ]; then
  TRAIN_EXTRA+=(--batch_size "${BATCH_SIZE}")
fi
if [ "${BADMINTON_RESUME_FOLDS:-1}" = "0" ] || [ "${BADMINTON_RESUME_FOLDS:-}" = "false" ] || [ "${BADMINTON_RESUME_FOLDS:-}" = "no" ]; then
  TRAIN_EXTRA+=(--no_resume_folds)
fi

train_one() {
  local m="$1"
  local out="${RUN_ROOT}/${m}"
  echo "=== [train] ${m} -> ${out} ==="
  python3 -m baseline train \
    --method "$m" \
    --loso_splits "$LOSO" \
    --run_dir "$out" \
    --fz_only \
    --save_report \
    --epochs 500 \
    --patience 80 \
    "${TRAIN_EXTRA[@]}" \
    2>&1 | tee "${LOG_DIR}/${m}.log"

  echo "=== [fuse] ${m} ==="
  python3 -m baseline fuse \
    --loso_splits "$LOSO" \
    --base_run_dir "$out" \
    --fz_only \
    --save_report \
    2>&1 | tee "${LOG_DIR}/${m}_fusion.log"
}

if [[ ! -f "$LOSO" ]]; then
  echo "Missing LOSO splits: $LOSO"
  exit 1
fi

if [[ "$MODE" == "train" && -n "$METHOD_ARG" ]]; then
  train_one "$METHOD_ARG"
  exit 0
fi

if [[ "$MODE" == "all" ]]; then
  for m in "${METHODS_FLAT[@]}"; do
    train_one "$m"
  done
  for m in "${METHODS_STGCN[@]}"; do
    train_one "$m"
  done
  echo ""
  echo "Paper bundle (canonical + wide table): python3 -m baseline paper-export --run-root $RUN_ROOT"
  python3 -m baseline paper-export --run-root "$RUN_ROOT"
  echo ""
  echo "Done. Outputs: $RUN_ROOT"
  echo "Optional single-view-only table: python3 -m baseline aggregate --inputs ${RUN_ROOT}/*/summary.json --out_md ${RUN_ROOT}/table.md --out_csv ${RUN_ROOT}/table.csv"
  exit 0
fi

echo "Usage: bash run_all_baselines.sh [all|train <method_id>]"
exit 1
