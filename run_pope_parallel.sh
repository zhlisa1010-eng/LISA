#!/bin/bash
cd /home/User/MLLMs/Qwen2.5vl_origin_backup

# ---------------- adversarial ----------------


/home/User/anaconda3/envs/deco_2.5/bin/python pope_single.py \
  --answers-file output/local/qwen2.5_caption_chair_beam_debug.jsonl \
  --decode-mode beam \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_adversarial.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &

# ---------------- popular ----------------
/home/User/anaconda3/envs/deco_2.5/bin/python pope_single_2.py \
  --answers-file output/POPE/popular/qwen2.5_caption_beam.jsonl \
  --decode-mode beam \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_popular.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &

/home/User/anaconda3/envs/deco_2.5/bin/python pope_single_7.py \
  --answers-file output/POPE/popular/qwen2.5_caption_chair_greedy.jsonl \
  --decode-mode greedy \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_popular.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &

/home/User/anaconda3/envs/deco_2.5/bin/python pope_single_0.py \
  --answers-file output/POPE/popular/qwen2.5_caption_chair_sample.jsonl \
  --decode-mode sample \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_popular.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &

# ---------------- random ----------------
/home/User/anaconda3/envs/deco_2.5/bin/python pope_single_5.py \
  --answers-file output/POPE/random/qwen2.5_caption_chair_beam.jsonl \
  --decode-mode beam \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_random.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &

/home/User/anaconda3/envs/deco_2.5/bin/python pope_single_7.py \
  --answers-file output/POPE/random/qwen2.5_caption_chair_greedy.jsonl \
  --decode-mode greedy \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_random.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &

/home/User/anaconda3/envs/deco_2.5/bin/python pope_single_1.py \
  --answers-file output/POPE/random/qwen2.5_caption_chair_sample.jsonl \
  --decode-mode sample \
  --question-file /home/User/MLLMs/data/POPE/coco_pope_random.jsonl \
  --image-folder /home/User/MLLMs/data/val2014 &
