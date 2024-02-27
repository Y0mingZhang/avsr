#!/bin/bash

ROOT_DIR="/data/datasets/hf_cache/tmp-lrs3/processed"
TEST_FILE="/data/datasets/hf_cache/tmp-lrs3/processed/labels/lrs3_test_transcript_lengths_seg24s.csv"


python eval.py data.modality=audio \
               data.dataset.root_dir=$ROOT_DIR \
               data.dataset.test_file=$TEST_FILE \
               pretrained_model_path="/data/datasets/hf_cache/tmp-lrs3/models/asr_trlrs3_base.pth"
