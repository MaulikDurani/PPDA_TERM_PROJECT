#!/bin/bash
# usage: bash run_grid_mezo.sh <your_mezo_script>.sh
SCRIPT=$1
LOGDIR=logs_2

SEEDS=(13 21 42 87 100)
TASKS=(SST-2 sst-5 SNLI MNLI RTE trec)

mkdir -p $LOGDIR

for SEED in "${SEEDS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    LOG=$LOGDIR/mezo_${TASK}_seed${SEED}.log

    # skip runs that already finished
    if [ -f "$LOG" ] && grep -q "Test results" "$LOG"; then
      echo "=== skipping task=$TASK seed=$SEED (already done) ==="
      continue
    fi

    echo "=== mezo seed=$SEED task=$TASK  $(date) ==="
    TASK=$TASK SEED=$SEED K=16 BS=8 STEP=10000 EVAL_STEP=10000 \
    bash $SCRIPT 2>&1 | tee $LOG
  done
done
echo "ALL DONE $(date)"
EOF
