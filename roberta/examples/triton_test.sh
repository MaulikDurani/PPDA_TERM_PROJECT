#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# DP-FZOO (sequential by default; optional batched forward), looped over TASKS x data SEEDS.
# Same settings and call as examples/fzoo_all_tasks.sh.
#
# Usage:  bash examples/triton_test.sh
#   TASKS="MNLI RTE" bash examples/triton_test.sh            (choose the tasks; several are allowed)
#   SEEDS="42" bash examples/triton_test.sh                  (one seed only)
#   FZOO_N=8 bash examples/triton_test.sh                      (more directions per step)
#   FZOO_PARALLEL=1 bash examples/triton_test.sh               (batched forward over the N directions; run tools/check_zo_parallel.py first)
#   STEP=50 EVAL_STEP=50 SEEDS="42" bash examples/triton_test.sh   (quick timing run)
#   ZO_FUSED=0 FZOO_PARALLEL=1 bash examples/triton_test.sh    (batched forward WITHOUT the fused Triton kernels, for A/B timing)
#   ZO_FAST_RNG=1 FZOO_PARALLEL=1 bash examples/triton_test.sh (also use the packed-bit Triton direction generator)
#   (ZO_FUSED / ZO_FAST_RNG only act on the batched path, i.e. with FZOO_PARALLEL=1)
#   FZOO_MOMENTUM=0.9 bash examples/triton_test.sh           (Nesterov momentum, 0 = off; works on both paths)
#   DRY_RUN=1 bash examples/triton_test.sh                   (print the plan and the checks, run nothing)
#   SUMMARY_ONLY=1 bash examples/triton_test.sh                (skip training, re-print the summary from the logs)
#
# Needs data/k-shot-1k-test/<TASK>/<K>-<SEED> for each task and seed (tools/generate_k_shot_data.py).
# Missing data directories are reported and skipped.

TASKS=${TASKS:-"SNLI MNLI RTE trec"}
SEEDS=${SEEDS:-"13 21 42 87 100"}
K=${K:-16}
BS=${BS:-32}
LR=${LR:-1e-4}
EPS=${EPS:-1e-4}
WD=${WD:-0}
STEP=${STEP:-10000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}
FZOO_N=${FZOO_N:-4}
FZOO_PARALLEL=${FZOO_PARALLEL:-0}   # 1 = batched forward over the N directions (src/zo_parallel.py)
PAR_FLAG=""
if [ "$FZOO_PARALLEL" = "1" ]; then PAR_FLAG="--fzoo_parallel"; fi

# Triton kernels in src/zo_parallel.py (read from the environment by ZOParallel; batched path only)
export ZO_FUSED=${ZO_FUSED:-1}          # 1 = fused perturbed-weight + update kernels, 0 = old PyTorch path
export ZO_FAST_RNG=${ZO_FAST_RNG:-0}    # 1 = packed-bit Triton direction generator (different random stream than 0)
if [ "$FZOO_PARALLEL" != "1" ]; then
    echo "note: FZOO_PARALLEL=0, so ZO_FUSED=$ZO_FUSED and ZO_FAST_RNG=$ZO_FAST_RNG have no effect (sequential path)"
fi
export FZOO_MOMENTUM=${FZOO_MOMENTUM:-0}   # Nesterov momentum beta in [0,1); 0 = off (read by src/trainer_fzoo.py)
if [ "$FZOO_PARALLEL" = "1" ]; then ZO_TAG=fused${ZO_FUSED}-rng${ZO_FAST_RNG}-mom${FZOO_MOMENTUM}; else ZO_TAG=seq-mom${FZOO_MOMENTUM}; fi

DPZERO_THRESHOLD=${DPZERO_THRESHOLD:-200.0}
DPZERO_PRIVACY_EPS=${DPZERO_PRIVACY_EPS:-2.0}
DPZERO_PRIVACY_DELTA=${DPZERO_PRIVACY_DELTA:-1e-5}

LOG_DIR=${LOG_DIR:-triton_BS32_E2_local}
mkdir -p $LOG_DIR

if [ "$SUMMARY_ONLY" != "1" ]; then
for TASK in $TASKS; do
for SEED in $SEEDS; do

    DATA=data/k-shot-1k-test/$TASK/$K-$SEED
    if [ ! -d "$DATA" ]; then
        echo "!!!!! skipping task=$TASK seed=$SEED: $DATA not found" | tee -a $LOG_DIR/skipped.txt
        continue
    fi

    GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP-n$FZOO_N-$ZO_TAG
    EXTRA_TAG=${EXTRA_TAG:-ft-}
    TAG=k${K}-${MODEL}-fzoo-dp-${EXTRA_TAG}
    echo "===== fzoo  task=$TASK  seed=$SEED  N=$FZOO_N  parallel=$FZOO_PARALLEL  $ZO_TAG ====="
    echo "Grid search tag: $GR_TAG"
    echo "Tag: $TAG"
    if [ "$DRY_RUN" = "1" ]; then continue; fi

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
        $PAR_FLAG \
        $@ 2>&1 | tee $LOG_DIR/${TASK}_seed${SEED}.log

    # the trainer logs its settings in one line; if it is missing, src/trainer_fzoo.py is an old version and the options above did nothing
    if ! grep -q "DP-FZOO config:" $LOG_DIR/${TASK}_seed${SEED}.log; then
        echo "!!!!! WARNING task=$TASK seed=$SEED: no 'DP-FZOO config:' line in the log (is src/trainer_fzoo.py the new version?)" | tee -a $LOG_DIR/skipped.txt
    fi
    if [ "$FZOO_PARALLEL" = "1" ] && ! grep -q "Parallel FZOO" $LOG_DIR/${TASK}_seed${SEED}.log; then
        echo "!!!!! WARNING task=$TASK seed=$SEED: no 'Parallel FZOO' line in the log, this run was probably SEQUENTIAL" | tee -a $LOG_DIR/skipped.txt
    fi

done
done
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo
echo "===== Summary (fzoo, K=$K, BS=$BS, LR=$LR, EPS=$EPS, eps_dp=$DPZERO_PRIVACY_EPS, N=$FZOO_N, parallel=$FZOO_PARALLEL, $ZO_TAG, $STEP steps) ====="
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
        DEV_ACC=$(awk '/Eval results/{f=1} f && /acc =/{print $NF; exit}' "$LOG")
        TEST_ACC=$(awk '/Test results/{f=1} f && /acc =/{print $NF; exit}' "$LOG")
        TEST_LOSS=$(awk '/Test results/{f=1} f && /eval_loss =/{print $NF; exit}' "$LOG")
        echo "task=$TASK seed=$SEED dev_acc=${DEV_ACC:-FAILED} test_acc=${TEST_ACC:-FAILED} test_loss=${TEST_LOSS:-FAILED}"
    done
done
} | tee $LOG_DIR/summary.txt

echo
echo "===== Mean / std of test accuracy per dataset ====="
for TASK in $TASKS; do
    grep "^task=$TASK " $LOG_DIR/summary.txt | grep -v "FAILED\|MISSING" | \
    awk -v t="$TASK" '{split($4,a,"="); n++; s+=a[2]; ss+=a[2]*a[2]}
         END {if (n>0) {m=s/n; v=(n>1)?(ss-n*m*m)/(n-1):0; sd=sqrt(v>0?v:0);
                        printf "task=%s  test_acc mean=%.4f  std=%.4f (sample)  stderr=%.4f  over %d seeds\n", t, m, sd, sd/sqrt(n), n}
              else printf "task=%s  no successful runs\n", t}'
done | tee -a $LOG_DIR/summary.txt
