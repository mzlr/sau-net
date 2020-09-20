#!/bin/bash

set -o pipefail

SPLIT=test

# evaluate baseline models
python tools/eval_classification_bn.py ${DATASET} ${SPLIT} ${MODEL} ${MODEL_PATH} ${SEED} \
    | tee -a ${MODEL_PATH}/log.txt