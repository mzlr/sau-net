#!/bin/bash

set -o pipefail

DISPLAY=10
SPLIT=train

mkdir -p ${MODEL_PATH};

# train model
python tools/train.py ${DATASET} ${SPLIT} ${MODEL} ${MODEL_PATH} ${SEED} \
    --iterations ${ITERATION_NUM} \
    --batch_size ${BATCH_SIZE} \
    --display ${DISPLAY} \
    --lr ${LEARNING_RATE} \
    --snapshot ${ITERATION_NUM} \
    --solver adamw | tee -a ${MODEL_PATH}/log.txt