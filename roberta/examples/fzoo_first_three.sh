#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# DP-FZOO on every dataset and every data seed (sequential FZOO, no batched/parallel forward).
# Same settings and call as examples/fzoo_quick.sh, looped over TASKS x SEEDS.
#
# Usage:  bash examples/fzoo_all_tasks.sh
#   TASKS="SST-2 RTE" SEEDS="42" bash examples/fzoo_all_tasks.sh    (subset)
#   FZOO_N=4 bash examples/fzoo_all_tasks.sh                         (fewer directions per step)
#   SUMMARY_ONLY=1 bash examples/fzoo_all_tasks.sh                   (skip training, re-print the summary from the logs)
#
# Needs data/k-shot-1k-test/<TASK>/16-<SEED> for each task and seed (tools/generate_k_shot_data.py).
# Missing data directories are reported and skipped instead of crashing the whole sweep.

TASKS=${TASKS:-"SST-2 sst-5 SNLI"}
SEEDS=${SEEDS:-"13 21 42 87 100"}
K=${K:-16}
BS=${BS:-8}
LR=${LR:-1e-4}
EPS=${EPS:-1e-4}
WD=${WD:-0}
STEP=${STEP:-10000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}
FZOO_N=${FZOO_N:-4}

DPZERO_THRESHOLD=${DPZERO_THRESHOLD:-200.0}
DPZERO_PRIVACY_EPS=${DPZERO_PRIVACY_EPS:-6.0}
DPZERO_PRIVACY_DELTA=${DPZERO_PRIVACY_DELTA:-1e-5}

LOG_DIR=${LOG_DIR:-logs_fzoo_first_three}
mkdir -p $LOG_DIR

if [ "$SUMMARY_ONLY" != "1" ]; then
for TASK in $TASKS; do
for SEED in $SEEDS; do

    DATA=data/k-shot-1k-test/$TASK/$K-$SEED
    if [ ! -d "$DATA" ]; then
        echo "!!!!! skipping task=$TASK seed=$SEED: $DATA not found" | tee -a $LOG_DIR/skipped.txt
        continue
    fi

    LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

    GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP-n$FZOO_N
    EXTRA_TAG=${EXTRA_TAG:-ft-}
    TAG=k${K}-${MODEL}-fzoo-dp-${EXTRA_TAG}
    echo "===== fzoo  task=$TASK  seed=$SEED ====="
    echo "Grid search tag: $GR_TAG"
    echo "Tag: $TAG"

    TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
        bash examples/run_fewshot_modified.sh \
        --per_device_train_batch_size $BS \
        --learning_rate $LR \
        --eval_steps $EVAL_STEP \
        --weight_decay $WD \
        --zero_order_eps $EPS \
        --zero_order_optim \
        --lr_scheduler_type "constant" \
        --optimizer "sgd" \
        --efficient_zero_order \
        --dpzero \
        --dpzero_clip_threshold $DPZERO_THRESHOLD \
        --dp_epsilon $DPZERO_PRIVACY_EPS \
        --dp_delta $DPZERO_PRIVACY_DELTA \
        --fzoo_n $FZOO_N \
        $@ 2>&1 | tee $LOG_DIR/${TASK}_seed${SEED}.log

done
done
fi

echo
echo "===== Summary (fzoo, K=$K, BS=$BS, LR=$LR, EPS=$EPS, eps_dp=$DPZERO_PRIVACY_EPS, N=$FZOO_N, $STEP steps) ====="
{
for TASK in $TASKS; do
    for SEED in $SEEDS; do
        LOG=$LOG_DIR/${TASK}_seed${SEED}.log
        if [ ! -f "$LOG" ]; then
            echo "task=$TASK seed=$SEED dev_acc=MISSING test_acc=MISSING test_loss=MISSING"
            continue
        fi
        # first 'eval_acc =' / 'eval_loss =' after "Eval results" is dev; after "Test results" is test
        DEV_ACC=$(awk '/Eval results/{f=1} f && /eval_acc =/{print $NF; exit}' $LOG)
        TEST_ACC=$(awk '/Test results/{f=1} f && /eval_acc =/{print $NF; exit}' $LOG)
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
         END {if (n>0) {m=s/n; v=(n>1)?(ss-n*m*m)/(n-1):0; printf "task=%s  test_acc mean=%.4f std=%.4f over %d seeds\n", t, m, sqrt(v>0?v:0), n}
              else printf "task=%s  no successful runs\n", t}'
done | tee -a $LOG_DIR/summary.txt