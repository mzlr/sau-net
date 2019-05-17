#!/bin/bash

# source scripts/config.sh

set -o pipefail

# SPLIT=${2:-test} # unset or null
SPLIT=test # unset or null

{
    # evaluate baseline models
    python tools/eval_classification_bn.py ${DATASET} ${SPLIT} ${MODEL} ${MODEL_PATH} ${SEED} \
        | tee -a ${MODEL_PATH}/log.txt &&

    cat ${MODEL_PATH}/log.txt | mail -s Job_completed yueguo3211@gmail.com
} || {
    cat ${MODEL_PATH}/log.txt | mail -s Job_failed yueguo3211@gmail.com
}
