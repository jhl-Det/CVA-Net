#!/usr/bin/env bash

set -x

EXP_DIR=./cache
PY_ARGS=${@:1}
ENC_LAYERS=6
DEC_LAYERS=6

python -u main.py \
    --output_dir ${EXP_DIR} \
    --enc_layers ${ENC_LAYERS} \
    --dec_layers ${DEC_LAYERS} \
    --data_mode '15frames' \
    --num_global_frames 3 \
    --num_feature_levels 1 \
    --batch_size 1 \
    --lr 5e-5 \
    --dist_url 'tcp://127.0.0.1:50001'
    ${PY_ARGS}
