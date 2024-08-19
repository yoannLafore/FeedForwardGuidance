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


# Example script, must be adapted to your needs

python ./fid_score.py \
  --pretrained_model_name_or_path "/path/to/model or model_name_huggingface" \
  --clip_model_name_for_guidance "openai/clip-vit-base-patch32" \
  --local_dataset_dir "/path/to/local_dataset" \
  --output_file "fid_score_normal.json" \
  --resolution 256 \
  --batch_size 1 \
  --clip_guidance_scale 100 \
  --caption_column "text" \
  --image_column "image" \
  --max_samples_to_select 6 \
  --seed 1498416156
