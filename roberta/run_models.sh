#!/bin/bash
SCRIPT=$1

# usage : bash run_models.sh examples/dpzero.sh
EPS_LIST=(2)
SEEDS=(13 21 42 87 100)
TASKS=(SST-2 sst-5 SNLI MNLI RTE trec)

for EPS_DP in "${EPS_LIST[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for TASK in "${TASKS[@]}"; do
      echo "=== eps=$EPS_DP seed=$SEED task=$TASK  $(date) ==="
      TASK=$TASK SEED=$SEED K=16 BS=8 STEP=10000 EVAL_STEP=10000 \
      DPZERO_PRIVACY_EPS=$EPS_DP DPZERO_PRIVACY_DELTA=1e-5 \
      bash $SCRIPT 2>&1 | tee logs/dpzero_${TASK}_eps${EPS_DP}_seed${SEED}.log
    done
  done
done
echo "ALL DONE $(date)"
EOF
