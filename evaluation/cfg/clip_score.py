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


import torch
import torch.utils.data
from diffusers import StableDiffusionPipeline
from datasets import load_from_disk, load_dataset
from tqdm.auto import tqdm
import argparse
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np
import json

def parse_args():
  parser = argparse.ArgumentParser(description="Clip score computing script")
  parser.add_argument(
      "--pretrained_model_name_or_path",
      type=str,
      default=None,
      required=True,
      help="Path to pretrained model or model identifier from huggingface.co/models.",
  )
  parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help=(
      "The name of the Dataset (from the HuggingFace hub) with the evalutation prompts (could be your own, possibly private,"
      " dataset)."
    ),
  )
  parser.add_argument(
    "--dataset_split",
    type=str,
    default="train",
    help=("The split of the dataset to use if it comes from huggingface hub")
  )
  parser.add_argument(
    "--local_dataset_dir",
    type=str,
    default=None,
    help=(
      "A folder containing a dataset created using the `datasets` library"
      "Ignored if `dataset_name` is specified"
    ),
  )
  parser.add_argument(
    "--dataset_config_name",
    type=str,
    default=None,
    help="The config of the Dataset, leave as None if there's only one config.",
  )
  parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help=("The file in which the clip score will be written (let None to only print it in the terminal)")
  )
  parser.add_argument(
    "--output_score_per_prompt_file",
    type=str,
    default=None,
    help=("The file in which the clip score for each prompt will be written (let None to not output the score per prompt)")
  )
  parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="The directory where the downloaded models and datasets will be stored.",
  )
  parser.add_argument(
    "--max_prompts_to_select",
    type=int,
    default=None,
    help=(
      "The maximum number of prompts used to compute the clip score"
    )
  )
  parser.add_argument(
    "--num_images_per_prompt",
    type=int,
    default=1,
    help=(
      "The number of images generated for each prompt (the average score will be taken for each prompt)"
    )
  )
  parser.add_argument(
    "--resolution",
    type=int,
    default=512,
    help=(
      "The resolution for the model images"
    ),
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help=(
      "Batch size for generating the images"
    )
  )
  parser.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help=(
      "Guidance scale used to generate the images (set to 1 to disable guidance)"
    )
  )
  parser.add_argument(
    "--prompt_column",
    type=str,
    default="text",
    help=(
      "The column of the input dataset containing the prompts"
    )
  )
  parser.add_argument(
    "--seed",
    type=int, 
    default=None, 
    help=(
      "A seed for reproducible clip score."
      )
    )

  args = parser.parse_args()

  if args.dataset_name is None and args.local_dataset_dir is None:
    raise ValueError("Need either a dataset name or a local dataset directory.")

  return args



def main():
  # Parse the arguments
  args = parse_args()

  if(args.seed is not None):
    torch.manual_seed(args.seed)

  # Get the device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load the pipeline for the model
  sd_pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None).to(device)

  

  # Load the dataset
  if args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir
    )

    dataset = dataset[args.dataset_split]
    
  elif args.local_dataset_dir is not None:
    dataset = load_from_disk(args.local_dataset_dir)
  else:
    raise ValueError(
      f"--dataset_name OR --local_dataset_dir must be specified"
    )

  # Truncate the dataset if necessary
  if(args.max_prompts_to_select is not None):
    dataset = dataset.shuffle(seed=args.seed).select(range(args.max_prompts_to_select))

  # Get the dataset column names
  column_names = dataset.column_names

  # Prompts
  prompt_column: str = args.prompt_column
  if prompt_column not in column_names:
    raise ValueError(
        f"--prompt_column' value '{args.prompt_column}' needs to be one of: {', '.join(column_names)}"
      )

  # Get the configuration arguments
  num_images_per_prompt: int = args.num_images_per_prompt
  image_size: int = args.resolution
  batch_size: int = args.batch_size
  guidance_scale: float = args.guidance_scale


  # Transform applied
  def transform(samples):
    prompts = [prompt for prompt in samples[prompt_column]]
    return {"prompts": prompts}
  
  # Create the dataloader
  dataset.set_transform(transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


  # Clip score function
  clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

  def calculate_clip_score(images: np.ndarray, prompts: list):
      scores = []
      
      for i in range(len(images)):
        current_images = images[i]
        current_prompts = [prompts[i]] * current_images.shape[0]
        images_int = (current_images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), current_prompts).detach()
        scores.append(float(clip_score))

      return np.array(scores)

  

  # Clip score computing loop
  progress_bar = tqdm(range(len(dataloader)))
  progress_bar.set_description("Clip score computing")

  all_scores = np.array([])

  for _, batch in enumerate(dataloader):
    prompts = batch["prompts"]
    current_batch_size = len(prompts)

    # Generate the images (silence the tqdm bar during the computation)
    sd_pipeline.set_progress_bar_config(disable=True)
    images: np.ndarray = sd_pipeline(prompts, height=image_size, width=image_size, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, output_type="numpy").images
    sd_pipeline.set_progress_bar_config(disable=False)
    
    images = images.reshape((current_batch_size, num_images_per_prompt, image_size, image_size, 3))

    current_scores = calculate_clip_score(images, prompts)
    all_scores = np.concatenate((all_scores, current_scores), axis=0)

    progress_bar.update(1)

  
  # Compute the final clip score
  final_clip_score = np.mean(all_scores)

  # Print the clip score
  print(f"Clip score : {final_clip_score}")

  # Save the outputs if necessary
  if(args.output_file is not None):
    # Create result report
    result_report = {
      "total_num_prompts": dataset.num_rows,
      "model": args.pretrained_model_name_or_path,
      "num_images_per_prompt": args.num_images_per_prompt,
      "dataset": args.dataset_name if args.dataset_name is not None else args.local_dataset_dir,
      "resolution": image_size,
      "batch_size": batch_size,
      "guidance_scale": guidance_scale,
      "seed": args.seed,
      "clip_score": final_clip_score
    }

    try:
      with open(args.output_file, 'w') as fp:
        json.dump(result_report, fp, indent=4)
        
    except IOError as err:
      print(f"Could not write '--output_file' {args.output_file} :\n {err}")

  # Save the raw clip scores (as per prompt score)
  if(args.output_score_per_prompt_file is not None):
    try:
      np.savetxt(args.output_score_per_prompt_file, all_scores)

    except IOError as err:
      print(f"Could save in '--output_score_per_prompt_file' {args.output_score_per_prompt_file} :\n {err}")


  
if __name__ == "__main__":
  main()
