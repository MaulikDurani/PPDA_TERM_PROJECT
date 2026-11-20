#!/bin/bash

# Runs fzoo_quick.sh (SST-2, K=16, BS=8, 500 steps) for every data seed and summarizes the test accuracy.
# Usage: bash examples/fzoo_seeds.sh        (override anything with env vars, e.g. STEP=500 LR=1e-6)

SEEDS=${SEEDS:-"13 21 42 87 100"}
STEP=${STEP:-500}
LOG_DIR=${LOG_DIR:-logs_fzoo_seeds}
mkdir -p $LOG_DIR

for S in $SEEDS; do
    echo "===== fzoo seed $S ====="
    SEED=$S STEP=$STEP EVAL_STEP=$STEP TASK=SST-2 K=16 BS=8 \
        bash examples/fzoo_quick.sh 2>&1 | tee $LOG_DIR/seed$S.log
done

echo
echo "===== Summary (fzoo, SST-2, K=16, BS=8, $STEP steps) ====="
for S in $SEEDS; do
    # the line "eval_acc = ..." that follows "Test results" is the test accuracy; the one after "Eval results" is dev
    DEV=$(awk '/Eval results/{f=1} f && /eval_acc =/{print $NF; exit}' $LOG_DIR/seed$S.log)
    TEST=$(awk '/Test results/{f=1} f && /eval_acc =/{print $NF; exit}' $LOG_DIR/seed$S.log)
    echo "seed $S  dev_acc=${DEV:-FAILED}  test_acc=${TEST:-FAILED}"
done | tee $LOG_DIR/summary.txt

awk '/test_acc=/ && !/FAILED/ {split($4,a,"="); n++; s+=a[2]; ss+=a[2]*a[2]}
     END {if (n>0) {m=s/n; v=(n>1)?(ss-n*m*m)/(n-1):0; printf "test_acc mean=%.4f std=%.4f over %d seeds\n", m, sqrt(v>0?v:0), n} else print "no successful runs"}' $LOG_DIR/summary.txt | tee -a $LOG_DIR/summary.txt