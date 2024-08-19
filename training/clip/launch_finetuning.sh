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

python ./finetuning.py \
  --pretrained_model_name_or_path "lambdalabs/miniSD-diffusers" \
  --local_dataset_dir "/path/to/local_dataset" \
  --output_dir "/path/to/output" \
  --noisy_latent_column "noisy_latent" \
  --noise_pred_column "noise_pred" \
  --clip_grad_column "clip_grad" \
  --timestep_column "timestep" \
  --caption_column "text" \
  --max_train_samples 20000 \
  --train_batch_size 8 \
  --num_train_epochs 60 \
  --guidance_scale_start 300 \
  --guidance_scale_end 300 \
  --checkpointing_steps 10000 \
  --checkpoints_total_limit 30 \
  --validation_prompts "Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet." \
  "Aerial photography of a winding river through autumn forests, with vibrant red and orange foliage." \
  "Fantasy illustration of a dragon perched on a castle, with a stormy sky and lightning in the background." \
  "Isometric digital art of a medieval village with thatched roofs, a market square, and townsfolk." \
  "Cute small cat sitting in a movie theater eating chicken wiggs watching a movie" \
  "realistic futuristic city-downtown with short buildings, sunset" \
  "seascape by Ray Collins and artgerm, front view of a perfect wave, sunny background, ultra detailed water" \
  "brown eyes, rutkowski repin, natural blonde hair realistic , image of an incredibly beautiful woman with highlights in her eyes" \
  --validation_epochs 1 \
  --seed 1234 \
  --resolution 256 \
  --report_to "wandb" \
  --tracker_project_name "miniSD_finetuning_clip" \
  --resume_from_checkpoint "latest" \
  --learning_rate 8e-7 \
  --lr_warmup_steps 0
  