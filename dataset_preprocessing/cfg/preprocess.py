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
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, SchedulerMixin
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import Dataset, Value, Features, Array3D, load_from_disk, load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import argparse


def parse_args():
  parser = argparse.ArgumentParser(description="Preprocessing script")
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
      "The name of the Dataset (from the HuggingFace hub) to preprocess (could be your own, possibly private,"
      " dataset)."
    ),
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
    "--cache_dir",
    type=str,
    default=None,
    help="The directory where the downloaded models and datasets will be stored.",
  )
  parser.add_argument(
    "--max_samples_to_preprocess",
    type=int,
    default=None,
    help=(
      "The maximum number of samples to be preprocessed"
    )
  )
  parser.add_argument(
    "--new_samples_per_input_sample",
    type=int,
    default=1,
    help=(
      "For one sample from the input dataset, how much new samples must be created"
    )
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help=(
      "Batch size for preprocessing the dataset"
    )
  )
  parser.add_argument(
    "--latent_dimension_sizes",
    type=int,
    nargs=3,
    default=[64, 64, 4],
    help="The dimension sizes of the latent space of the images (must be 3 dimensional)"

  )
  parser.add_argument(
    "--output_dir",
    type=str,
    default="sd-dataset-preprocessed",
    help=(
      "The directory in which the preprocessed dataset will be saved"
      )
  )
  parser.add_argument(
    "--output_dataset_name",
    type=str,
    default="preprocessed_latents",
    help=(
      "The name of the newly created dataset after preprocessing"
    )
  )
  parser.add_argument(
    "--input_image_column",
    type=str,
    default="image",
    help=(
      "The column of the input dataset containing the images"
    )
  )
  parser.add_argument(
    "--input_caption_column",
    type=str,
    default="text",
    help=(
      "The column of the input dataset containing the captions"
    )
  )
  parser.add_argument(
    "--output_noisy_latent_column", 
    type=str, 
    default="noisy_image", 
    help="The column of the output preprocessed dataset containing an the noisy latent image."
  )
  parser.add_argument(
    "--output_noise_pred_uncond_column", 
    type=str, 
    default="noise_pred_uncond", 
    help="The column of the output preprocessed dataset containing the unconditional prediction of the noise"
  )
  parser.add_argument(
    "--output_noise_pred_cond_column", 
    type=str, 
    default="noise_pred_cond", 
    help="The column of the output preprocessed dataset containing the conditional prediction of the noise"
  )
  parser.add_argument(
    "--output_timestep_column",
    type=str,
    default="timestep", 
    help="The column of the output preprocessed dataset containing the timestep of the corresponding noisy latent"
  )
  parser.add_argument(
    "--output_caption_column",
    type=str,
    default="text",
    help="The column of the output preprocessed dataset containing a caption or a list of captions.",
  )
  parser.add_argument(
    "--resolution",
    type=int,
    default=512,
    help=(
      "The resolution for input images, all the images in the train/validation dataset will be resized to this"
      " resolution"
    ),
  )
  parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible preprocessing.")


  args = parser.parse_args()

  if args.dataset_name is None and args.local_dataset_dir is None:
    raise ValueError("Need either a dataset name or a local dataset directory.")
  
  return args



def main():
  # Parse the arguments
  args = parse_args()

  if(args.seed is not None):
    torch.manual_seed(args.seed)

  # Load the pretrained model
  vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
  tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
  text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
  unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
  scheduler: PNDMScheduler = PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") 

  # Get the device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  vae.to(device)
  text_encoder.to(device)
  unet.to(device)

  # Load the dataset
  if args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir
    )

    dataset = dataset["train"]
    
  elif args.local_dataset_dir is not None:
    dataset = load_from_disk(args.local_dataset_dir)
  else:
    raise ValueError(
      f"--dataset_name OR --local_dataset_dir must be specified"
    )

  # Truncate the dataset if necessary
  if(args.max_samples_to_preprocess is not None):
    dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples_to_preprocess))


  # Get the input dataset column names for image/text
  column_names = dataset.column_names

  # Images
  image_column: str = args.input_image_column
  if image_column not in column_names:
    raise ValueError(
        f"--input_image_column' value '{args.input_image_column}' needs to be one of: {', '.join(column_names)}"
      )
  
  # Captions
  caption_column: str = args.input_caption_column
  if caption_column not in column_names:
    raise ValueError(
        f"--input_caption_column' value '{args.input_caption_column}' needs to be one of: {', '.join(column_names)}"
      )

  # Get the configuration arguments
  image_size: int = args.resolution
  batch_size: int = args.batch_size
  num_timesteps: int = scheduler.config.num_train_timesteps
  num_samples_per_image: int = args.new_samples_per_input_sample
  latent_dimension_sizes: tuple = tuple(args.latent_dimension_sizes)

  output_noisy_latent_column: str = args.output_noisy_latent_column
  output_noise_pred_uncond_column: str = args.output_noise_pred_uncond_column
  output_noise_pred_cond_column: str = args.output_noise_pred_cond_column
  output_timestep_column: str = args.output_timestep_column
  output_caption_column: str = args.output_caption_column

  output_dir: str = args.output_dir
  output_dataset_name: str = args.output_dataset_name

  # Pre-preprossecing functions applied to the input dataset
  preprocess_transforms = transforms.Compose(
    [
      transforms.Resize((image_size, image_size)),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ]
  )

  # Transform applied to the input dataset
  def transform(samples):
    images = [preprocess_transforms(image.convert("RGB")) for image in samples[image_column]]
    labels = [label for label in samples[caption_column]]
    return {"images": images, "labels": labels}

  # Create the dataloader
  dataset.set_transform(transform)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)


  # Features of the new dataset
  features = Features({
    output_caption_column: Value("string"),
    output_timestep_column: Value("int32"),
    output_noisy_latent_column: Array3D(shape=latent_dimension_sizes, dtype="float32"),
    output_noise_pred_cond_column: Array3D(shape=latent_dimension_sizes, dtype="float32"),
    output_noise_pred_uncond_column: Array3D(shape=latent_dimension_sizes, dtype="float32"),
  })

  # Main preprocessing loop, generator for the new dataset
  def processing_loop_gen():
    progress_bar = tqdm(range(len(dataloader) * num_samples_per_image))
    progress_bar.set_description("Preprocessing")

    for sample_i in range(num_samples_per_image):

      for _, batch in enumerate(dataloader):
        clean_images = batch["images"].to(device)
        prompts = batch["labels"]
        current_batch_size = clean_images.shape[0]

        with torch.no_grad():
          # Create embeddings for the prompts
          prompts_inputs = tokenizer(
            prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
          )
          prompts_embeddings = text_encoder(
            prompts_inputs.input_ids.to(device)
          )[0]
          max_length = prompts_inputs.input_ids.shape[-1]

          # Create unconditional embeddings
          uncond_input = tokenizer([""] * current_batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
          uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]


          # Encode the input images
          clean_images_latent = vae.encode(clean_images).latent_dist.mode()

          # Add noise to the clean images for random timesteps
          noise = torch.randn(clean_images_latent.shape).to(device)
          timesteps = torch.randint(
            0, num_timesteps, (current_batch_size, ), device=device
          )
          noisy_latents = scheduler.add_noise(clean_images_latent, noise, timesteps)


          # Predict the noise using the model
          noise_pred_cond = unet(noisy_latents, timesteps, encoder_hidden_states=prompts_embeddings).sample
          noise_pred_uncond = unet(noisy_latents, timesteps, encoder_hidden_states=uncond_embeddings).sample


          # Yield the items to add to the dataset
          for i in range(current_batch_size):
            item = {
              output_caption_column: prompts[i],
              output_timestep_column: timesteps[i],
              output_noisy_latent_column: noisy_latents[i],
              output_noise_pred_cond_column: noise_pred_cond[i],
              output_noise_pred_uncond_column: noise_pred_uncond[i]
              }
            yield item

          progress_bar.update(1)

  # Compute and save the new dataset
  new_dataset = Dataset.from_generator(generator=lambda: processing_loop_gen(), features=features, dataset_name=output_dataset_name)
  new_dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    main()