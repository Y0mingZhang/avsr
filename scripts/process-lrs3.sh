#!/bin/bash

DATA_DIR="/data/datasets/hf_cache/tmp-lrs3"
ROOT_DIR="/data/datasets/hf_cache/tmp-lrs3/processed-torchaudio"

# for i in {0..63}
# do
# sbatch scripts/single-cpu.sh python data_prep/preprocess_lrs3.py \
#     --data-dir=$DATA_DIR \
#     --dataset="lrs3" \
#     --root-dir=$ROOT_DIR \
#     --subset="train" \
#     --groups=64 \
#     --job-index=$i
# done

sbatch scripts/single-cpu.sh python data_prep/preprocess_lrs3.py \
    --data-dir=$DATA_DIR \
    --dataset="lrs3" \
    --root-dir=$ROOT_DIR \
    --subset="test" \
    --groups=1


python data_prep/merge.py \
    --root-dir=$ROOT_DIR \
    --dataset=lrs3 \
    --subset=train \
    --groups=64