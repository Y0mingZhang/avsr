#!/bin/bash

ROOT_DIR="/data/datasets/hf_cache/tmp-lrs3/processed-torchaudio"

sbatch scripts/disubmit.sh python train.py \
                --exp-dir="avsr" \
                --exp-name="avsr" \
                --modality="audiovisual" \
                --mode="offline" \
                --root-dir=$ROOT_DIR \
                --sp-model-path="spm_unigram_1023.model" \
                --num-nodes=1 \
                --gpus=2

sbatch scripts/disubmit.sh python train.py \
                --exp-dir="additive-avsr" \
                --exp-name="additive-avsr" \
                --modality="audiovisual" \
                --architecture "additive-av-conformer" \
                --mode="offline" \
                --root-dir=$ROOT_DIR \
                --sp-model-path="spm_unigram_1023.model" \
                --num-nodes=1 \
                --gpus=2

sbatch scripts/disubmit.sh python train.py \
                --exp-dir="voting-avsr" \
                --exp-name="voting-avsr" \
                --modality="audiovisual" \
                --architecture "voting-av-conformer" \
                --mode="offline" \
                --root-dir=$ROOT_DIR \
                --sp-model-path="spm_unigram_1023.model" \
                --num-nodes=1 \
                --gpus=2

