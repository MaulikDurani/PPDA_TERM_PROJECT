#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Quick sanity run: SST-2 only, K=16 shots per class, batch size 8, few steps (target: ~30 minutes)
TASK=${TASK:-SST-2}
K=${K:-16}
SEED=${SEED:-42}
BS=${BS:-32}
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

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP-n$FZOO_N
EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-fzoo-dp-${EXTRA_TAG}}
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
    $@