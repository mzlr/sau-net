#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH
export MODEL=unet
export DATASET=${1:-vgg} # unset or null #adi #vgg #mbm # vgg
export ITERATION_NUM=${2:-350} # unset or null
export LEARNING_RATE=0.001

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


for i in {1..10}
do
    export SEED=$RANDOM #18031 #123232
    export MODEL_PATH=snapshot/${MODEL}_${DATASET}_${ITERATION_NUM}_${SEED}
    scripts/train.sh
    scripts/test_bn.sh
done

