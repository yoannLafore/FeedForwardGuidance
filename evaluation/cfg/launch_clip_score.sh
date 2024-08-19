#!/bin/sh

# MIT License
# Copyright (c) 2024 Yoann Lafore
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Sample script, adjust to your needs

python ./clip_score.py \
  --pretrained_model_name_or_path "/path/to/model or model_name_huggingface" \
  --dataset_name "dataset_name_huggingface" \
  --output_file "clip_score_finetuned.json" \
  --output_score_per_prompt_file "raw_scores_finetuned.out" \
  --num_images_per_prompt 3 \
  --resolution 256 \
  --batch_size 8 \
  --guidance_scale 1 \
  --prompt_column "Prompts" \
  --max_prompts_to_select 200 \
  --seed 1498416156



# Example with miniSD and Drawbench

# python ./clip_score.py \
#   --pretrained_model_name_or_path "lambdalabs/miniSD-diffusers" \
#   --dataset_name "sayakpaul/drawbench" \
#   --output_file "score_normal_guidance.json" \
#   --output_score_per_prompt_file "raw_scores_normal_guidance.out" \
#   --num_images_per_prompt 3 \
#   --resolution 256 \
#   --batch_size 8 \
#   --guidance_scale 7.5 \
#   --prompt_column "Prompts" \
#   --max_prompts_to_select 200 \
#   --seed 1498416156