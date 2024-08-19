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


# Example script, must be modify to your needs

python ./preprocess.py \
  --pretrained_model_name_or_path "lambdalabs/miniSD-diffusers" \
  --local_dataset_dir "/path/to/local_dataset" \
  --max_samples_to_preprocess 5000 \
  --new_samples_per_input_sample 1 \
  --batch_size 6 \
  --latent_dimension_sizes 4 32 32 \
  --blue_channel_target 0.9 \
  --output_dir "sd-dataset-preprocessed-blue" \
  --output_dataset_name "blue-preprocessed" \
  --input_image_column "image" \
  --input_caption_column "text" \
  --output_noisy_latent_column "noisy_latent" \
  --output_noise_pred_column "noise_pred" \
  --output_blue_grad_column "blue_grad" \
  --output_timestep_column "timestep" \
  --output_caption_column "text" \
  --resolution 256 \
  --seed 1234 \
  