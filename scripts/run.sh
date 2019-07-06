#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export MODEL=unet
export DATASET=${1:-vgg} # unset or null #adi #vgg #mbm # vgg
export ITERATION_NUM=${2:-350} # unset or null
export LEARNING_RATE=0.001
RUN=${3:-1}

if [ "$DATASET" = "vgg" ]; then
    export BATCH_SIZE=75
elif [ "$DATASET" == "adi" ]; then
    export BATCH_SIZE=75
elif [ "$DATASET" == "mbm" ]; then
    export BATCH_SIZE=15
elif [ "$DATASET" == "dcc" ]; then
    export BATCH_SIZE=75
else
	echo "Incorrect dataset"
    exit 1
fi


for i in $(seq 1 $RUN)
do
    export SEED=$RANDOM
    export MODEL_PATH=snapshot/${MODEL}_${DATASET}_${ITERATION_NUM}_${SEED}
    scripts/train.sh
    scripts/test_bn.sh
done

