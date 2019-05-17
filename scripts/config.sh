#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export MODEL=unet
export DATASET=${1:-vgg} # unset or null #adi #vgg #mbm # vgg

if [ "$DATASET" = "vgg" ]; then
	export ITERATION_NUM=1550
    export BATCH_SIZE=75
    export LEARNING_RATE=0.01
elif [ "$DATASET" == "adi" ]; then
	export ITERATION_NUM=1550
    export BATCH_SIZE=200
    export LEARNING_RATE=0.01
elif [ "$DATASET" == "mbm" ]; then
	export ITERATION_NUM=750
    export BATCH_SIZE=15
    export LEARNING_RATE=0.001
else
	echo "Incorrect dataset"
    exit 1
fi

export SEED=123232 #18031 #123232
export MODEL_PATH=snapshot/${MODEL}_${DATASET}_${ITERATION_NUM} #_$(hostname)