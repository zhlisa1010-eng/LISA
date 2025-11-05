#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lisa_env

cd /home/User/MLLMs/Qwen2.5vl_origin

# 1. greedy

echo "===== Running greedy mode ====="
python local.py --answers-file output/local/qwen2.5_caption_chair_greedy.jsonl --decode-mode greedy

# python local.py --answers-file output/local/qwen2.5_caption_chair_greedy_test.jsonl --decode-mode greedy
echo "===== greedy mode finished ====="

# 2. beam search

echo "===== Running beam mode ====="
python local.py --answers-file output/local/qwen2.5_caption_chair_beam.jsonl --decode-mode beam --num_beams 5

echo "===== beam mode finished ====="

# 3. sample

echo "===== Running sample mode ====="
python local.py --answers-file output/local/qwen2.5_caption_chair_sample.jsonl --decode-mode sample --temperature 0.7 --top-p 0.9

echo "===== sample mode finished ====="

# 4. dola

# echo "===== Running dola mode ====="
# python local.py --answers-file output/local/qwen2.5_caption_chair_dola.jsonl --decode-mode dola --start-layer 20 --end-layer 28

# echo "===== dola mode finished =====" 