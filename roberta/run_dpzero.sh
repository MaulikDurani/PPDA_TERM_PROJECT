#!/bin/bash

# DPZero (examples/dpzero.sh) on every TASK x data SEED, with logging. Same layout as examples/fzoo_parallel_all.sh.
#
# Usage:  bash run_models.sh examples/dpzero.sh
#   DRY_RUN=1 bash run_models.sh examples/dpzero.sh                (print the plan and the checks, run nothing)
#   TASKS="SST-2" SEEDS="42" bash run_models.sh examples/dpzero.sh (subset)
#   DPZERO_PRIVACY_EPS=6.0 bash run_models.sh examples/dpzero.sh   (another privacy budget)
#   BS=32 LR=1e-6 bash run_models.sh examples/dpzero.sh            (other batch size / learning rate)
#   STEP=50 EVAL_STEP=50 SEEDS="42" bash run_models.sh examples/dpzero.sh   (quick timing run)
#   SUMMARY_ONLY=1 bash run_models.sh examples/dpzero.sh           (skip training, re-print the summary from the logs)
#
# Logs: $LOG_DIR/<TASK>_seed<SEED>.log, summary in $LOG_DIR/summary.txt.
# Needs data/k-shot-1k-test/<TASK>/<K>-<SEED> for each task and seed (tools/generate_k_shot_data.py).
# Missing data directories are reported and skipped.

SCRIPT=${1:-examples/dpzero.sh}

TASKS=${TASKS:-"SST-2 sst-5 SNLI"}
SEEDS=${SEEDS:-"13 21 42 87 100"}
K=${K:-16}
BS=${BS:-32}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}          # zero-order probe size (lambda), read by examples/dpzero.sh
WD=${WD:-0}
STEP=${STEP:-10000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}

DPZERO_THRESHOLD=${DPZERO_THRESHOLD:-200.0}
DPZERO_PRIVACY_EPS=${DPZERO_PRIVACY_EPS:-2.0}
DPZERO_PRIVACY_DELTA=${DPZERO_PRIVACY_DELTA:-1e-5}

LOG_DIR=${LOG_DIR:-dpzero_eps${DPZERO_PRIVACY_EPS}_bs${BS}_lr${LR}}
mkdir -p $LOG_DIR

if [ "$SUMMARY_ONLY" != "1" ]; then
for TASK in $TASKS; do
for SEED in $SEEDS; do

    DATA=data/k-shot-1k-test/$TASK/$K-$SEED
    if [ ! -d "$DATA" ]; then
        echo "!!!!! skipping task=$TASK seed=$SEED: $DATA not found" | tee -a $LOG_DIR/skipped.txt
        continue
    fi
    echo "===== dpzero  task=$TASK  seed=$SEED  BS=$BS  LR=$LR  lambda=$EPS  eps_dp=$DPZERO_PRIVACY_EPS  $(date) ====="
    if [ "$DRY_RUN" = "1" ]; then continue; fi

    TASK=$TASK SEED=$SEED K=$K BS=$BS LR=$LR EPS=$EPS WD=$WD STEP=$STEP EVAL_STEP=$EVAL_STEP MODEL=$MODEL \
    DPZERO_THRESHOLD=$DPZERO_THRESHOLD DPZERO_PRIVACY_EPS=$DPZERO_PRIVACY_EPS DPZERO_PRIVACY_DELTA=$DPZERO_PRIVACY_DELTA \
        bash $SCRIPT 2>&1 | tee $LOG_DIR/${TASK}_seed${SEED}.log

    # the trainer prints its arguments at the top of the log; if the privacy budget in the log is not the one asked for, the flag did not take effect
    LOGGED_EPS=$(grep -m1 "^dp_epsilon=" $LOG_DIR/${TASK}_seed${SEED}.log | sed 's/dp_epsilon=\([0-9.eE+-]*\).*/\1/')
    if [ -z "$LOGGED_EPS" ] || ! awk -v a="$LOGGED_EPS" -v b="$DPZERO_PRIVACY_EPS" 'BEGIN{exit !(a+0==b+0)}'; then
        echo "!!!!! WARNING task=$TASK seed=$SEED: log says dp_epsilon=${LOGGED_EPS:-MISSING}, expected $DPZERO_PRIVACY_EPS (is $SCRIPT reading DPZERO_PRIVACY_EPS?)" | tee -a $LOG_DIR/skipped.txt
    fi

done
done
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo
echo "===== Summary (dpzero, K=$K, BS=$BS, LR=$LR, lambda=$EPS, eps_dp=$DPZERO_PRIVACY_EPS, $STEP steps) ====="
{
for TASK in $TASKS; do
    for SEED in $SEEDS; do
        LOG=$LOG_DIR/${TASK}_seed${SEED}.log
        if [ ! -f "$LOG" ]; then
            echo "task=$TASK seed=$SEED dev_acc=MISSING test_acc=MISSING test_loss=MISSING"
            continue
        fi
        # metric key differs per task (eval_acc, eval_snli/acc, ...): match any "...acc ="
        # first match after "Eval results" is dev; after "Test results" is test
        DEV_ACC=$(awk '/Eval results/{f=1} f && /acc =/{print $NF; exit}' $LOG)
        TEST_ACC=$(awk '/Test results/{f=1} f && /acc =/{print $NF; exit}' $LOG)
        TEST_LOSS=$(awk '/Test results/{f=1} f && /eval_loss =/{print $NF; exit}' $LOG)
        echo "task=$TASK seed=$SEED dev_acc=${DEV_ACC:-FAILED} test_acc=${TEST_ACC:-FAILED} test_loss=${TEST_LOSS:-FAILED}"
    done
done
} | tee $LOG_DIR/summary.txt

echo
echo "===== Mean / std of test accuracy per dataset ====="
for TASK in $TASKS; do
    grep "^task=$TASK " $LOG_DIR/summary.txt | grep -v "FAILED\|MISSING" | \
    awk -v t=$TASK '{split($4,a,"="); n++; s+=a[2]; ss+=a[2]*a[2]}
         END {if (n>0) {m=s/n; v=(n>1)?(ss-n*m*m)/(n-1):0; sd=sqrt(v>0?v:0);
                        printf "task=%s  test_acc mean=%.4f  std=%.4f (sample)  stderr=%.4f  over %d seeds\n", t, m, sd, sd/sqrt(n), n}
              else printf "task=%s  no successful runs\n", t}'
done | tee -a $LOG_DIR/summary.txt
