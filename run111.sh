#!/usr/bin/env bash

COMPUTER="root"


PY=python


CUDA_VISIBLE_DEVICES=$1 $PY main.py --dataset $2 \
        --window-size 4 --neg-ratio 3 --embedding-dim 256 \
        --lstm-layers 1 --epoch-number 100  --batch-size 200 \
        --learning-rate 0.003 --cnn-channel 64 --lambda 100000 \
        --length $5 --coverage $4 \
        --precision_at_K 5 --id $3 --test-ratio 0.1 \
        --include-content 1
        # --gen-metapaths --length 15 --coverage 3 --alpha 0.0 --metapaths "AQRQA" 
        # --preprocess --test-threshold 3 --proportion-test 0.1 --test-size 20\


