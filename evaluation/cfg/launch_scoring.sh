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


# Launch scoring for a list of guidance scales
# Must be adapted to your needs

guidance_scales="1.0 1.5 2.0 3.0 4.0 5.0 6.0 7.0 7.5 8.0"

mkdir -p scores
mkdir -p scores/clip scores/fid

for g in $guidance_scales
do
  python ./clip_score.py \
    --pretrained_model_name_or_path "/path/to/model or model_name_huggingface" \
    --dataset_name "sayakpaul/drawbench" \
    --output_file "scores/clip/clip_${g}.json" \
    --num_images_per_prompt 3 \
    --resolution 256 \
    --batch_size 8 \
    --guidance_scale $g \
    --prompt_column "Prompts" \
    --max_prompts_to_select 200 \
    --seed 1498416156

python ./fid_score.py \
  --pretrained_model_name_or_path "/path/to/model or model_name_huggingface" \
  --local_dataset_dir "/path/to/dataset or dataset_name_huggingface" \
  --output_file "scores/fid/fid_${g}.json" \
  --resolution 256 \
  --batch_size 8 \
  --guidance_scale $g \
  --caption_column "text" \
  --image_column "image" \
  --max_samples_to_select 600 \
  --seed 1498416156
done