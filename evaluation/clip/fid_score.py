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
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from datasets import load_from_disk, load_dataset
from tqdm.auto import tqdm
import argparse
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
import json
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPImageProcessor
from clip_guided_stable_diffusion import CLIPGuidedStableDiffusion


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
    "--clip_model_name_for_guidance",
    type=str,
    default="openai/clip-vit-base-patch32",
    help="CLIP model used to perform the inference CLIP guidance"
  )
  parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help=(
      "The name of the Dataset (from the HuggingFace hub) with the evalutation prompts/images (could be your own, possibly private,"
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
    help=("The file in which the fid score will be written (let None to only print it in the terminal)")
  )
  parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="The directory where the downloaded models and datasets will be stored.",
  )
  parser.add_argument(
    "--max_samples_to_select",
    type=int,
    default=None,
    help=(
      "The maximum number of samples used to compute the fid score"
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
    "--clip_guidance_scale",
    type=float,
    default=100,
    help=(
      "CLIP Guidance scale used to generate the images (set to 0 to disable guidance)"
    )
  )
  parser.add_argument(
    "--caption_column",
    type=str,
    default="text",
    help=(
      "The column of the input dataset containing the captions"
    )
  )
  parser.add_argument(
    "--image_column",
    type=str,
    default="image",
    help=(
      "The column of the input dataset containing the images"
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

  sd_model_name: str = args.pretrained_model_name_or_path
  clip_model_name = args.clip_model_name_for_guidance

  # Load the pipeline for the model
  vae: AutoencoderKL = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae")
  tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")
  text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder")
  unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet")
  scheduler: PNDMScheduler = PNDMScheduler.from_pretrained(sd_model_name, subfolder="scheduler")
  clip_model: CLIPModel = CLIPModel.from_pretrained(clip_model_name)
  feature_extractor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(clip_model_name)

  vae.to(device)
  text_encoder.to(device)
  unet.to(device)
  clip_model.to(device)

  sd_pipeline : CLIPGuidedStableDiffusion = CLIPGuidedStableDiffusion(vae, text_encoder, clip_model, tokenizer, unet, scheduler, feature_extractor)
  sd_pipeline.to(device)
  

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
  if(args.max_samples_to_select is not None):
    dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples_to_select))

  # Get the dataset column names
  column_names = dataset.column_names

  # Captions
  caption_column: str = args.caption_column
  if caption_column not in column_names:
    raise ValueError(
        f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
      )

  # Images
  image_column: str = args.image_column
  if image_column not in column_names:
    raise ValueError(
        f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
      )

  # Get the configuration arguments
  image_size: int = args.resolution
  batch_size: int = args.batch_size
  clip_guidance_scale: float = args.clip_guidance_scale

  # Pre-preprossecing functions applied to the input dataset
  def preprocess_image(image):
      image = torch.tensor(image).unsqueeze(0)
      image = image.permute(0, 3, 1, 2) / 255.0
      return F.center_crop(image, (256, 256))[0]


  def transform(samples):
    captions = [caption for caption in samples[caption_column]]
    images = [preprocess_image(np.array(image.convert("RGB"))) for image in samples[image_column]]
    return {"images": images, "captions": captions}


  dataset.set_transform(transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


  fid = FrechetInceptionDistance(normalize=True)


  # FID score computing loop
  progress_bar = tqdm(range(len(dataloader)))
  progress_bar.set_description("FID score computing")

  for _, batch in enumerate(dataloader):
    captions = batch["captions"]
    real_images = batch['images']
    current_batch_size = len(captions)

    # Generate the images (silence the tqdm bar during the computation)
    sd_pipeline.set_progress_bar_config(disable=True)
    fake_images = []

    # Note : The CLIPGuidedDiffusionPipeline crashes if more than ones prompts is given at a time
    # This is why the full batch is not computed all in one go
    for caption in captions:

      fake_image: np.ndarray = sd_pipeline(
        caption,
        height=image_size,
        width=image_size,
        guidance_scale=1,
        use_cutouts=False,
        clip_prompt=None,
        clip_guidance_scale=clip_guidance_scale,
        num_images_per_prompt=1,
        output_type="numpy").images[0]
      
      fake_images.append(fake_image)

    fake_images = np.array(fake_images)

    sd_pipeline.set_progress_bar_config(disable=False)
    
    # Preprocess the fake images
    fake_images = torch.tensor(fake_images)
    fake_images = fake_images.permute(0, 3, 1, 2)

    # Add the real/fake images to the FID computation
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    progress_bar.update(1)


  fid_score = float(fid.compute())

  print(f"FID score : {fid_score}")

  # Save the outputs if necessary
  if(args.output_file is not None):
    # Create result report
    result_report = {
      "total_num_samples": dataset.num_rows,
      "model": args.pretrained_model_name_or_path,
      "clip_model": args.clip_model_name_for_guidance,
      "dataset": args.dataset_name if args.dataset_name is not None else args.local_dataset_dir,
      "resolution": image_size,
      "batch_size": batch_size,
      "clip_guidance_scale": clip_guidance_scale,
      "seed": args.seed,
      "fid_score": fid_score
    }

    try:
      with open(args.output_file, 'w') as fp:
        json.dump(result_report, fp, indent=4)
        
    except IOError as err:
      print(f"Could not write '--output_file' {args.output_file} :\n {err}")



if __name__ == "__main__":
  main()