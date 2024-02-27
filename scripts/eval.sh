#!/bin/bash

ROOT_DIR="/data/datasets/hf_cache/tmp-lrs3/processed-torchaudio"
TEST_FILE="/data/datasets/hf_cache/tmp-lrs3/processed-torchaudio/labels/lrs3_test_transcript_lengths_seg16s.csv"


python eval.py --modality="audiovisual" \
               --architecture="voting-av-conformer" \
               --mode="offline" \
               --root-dir=$ROOT_DIR \
               --sp-model-path=spm_unigram_1023.model \
               --checkpoint-path="voting-avsr/voting-avsr/model_avg_10.pth" &

python eval.py --modality="audiovisual" \
               --architecture="additive-av-conformer" \
               --mode="offline" \
               --root-dir=$ROOT_DIR \
               --sp-model-path=spm_unigram_1023.model \
               --checkpoint-path="additive-avsr/additive-avsr/model_avg_10.pth" &

wait
