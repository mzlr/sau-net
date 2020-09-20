#!/bin/bash
set -e

DATASET=${1:-1}
SELF_ATTN=${2:-1}
GPU_ID=${3:-0}

OUTPUT_DIR=outputs/${DATASET}
[ $SELF_ATTN -eq 1 ] && OUTPUT_DIR+="_sattn"

RNG_SEED=$RANDOM
OUTPUT_DIR+="_${RNG_SEED}"

BATCH_SIZE=16

STEP=12750
MOMENTUM=0.9
PATCH_SIZE=128
PATCH_DEPTH=32

mkdir -p ${OUTPUT_DIR}

python train_3d.py \
DATASET ${DATASET} \
OUTPUT_DIR ${OUTPUT_DIR} \
TRAIN.STEP ${STEP} \
MODEL.BN_MOMENTUM ${MOMENTUM} \
GPU_ID ${GPU_ID} \
TRAIN.BATCH_SIZE ${BATCH_SIZE} \
TRAIN.PATCH_SIZE ${PATCH_SIZE} \
TRAIN.PATCH_DEPTH ${PATCH_DEPTH} \
RNG_SEED ${RNG_SEED} \
SELF_ATTN ${SELF_ATTN} \
2>&1 | tee -a ${OUTPUT_DIR}/log.txt
