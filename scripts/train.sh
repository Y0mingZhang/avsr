#!/bin/bash

ROOT_DIR="/data/datasets/hf_cache/tmp-lrs3/processed-torchaudio"

python train.py --exp-dir="avsr-offline" \
                --exp-name="avsr-offline" \
                --modality="audiovisual" \
                --mode="offline" \
                --root-dir=$ROOT_DIR \
                --sp-model-path="spm_unigram_1023.model" \
                --num-nodes=1 \
                --gpus=2