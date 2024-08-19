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


from datasets import load_dataset, Dataset

from dataclasses import dataclass
from datasets import load_dataset, Dataset
from PIL import Image

import aiohttp
from PIL import Image
import asyncio
from aiohttp import ClientSession, ClientTimeout
from io import BytesIO

@dataclass
class DownloadConfig:
  dataset_to_download_name = "ChristophSchuhmann/improved_aesthetics_6.25plus"
  num_entries = 1_000_000 # Number of entries that will be tried to be downloaded
  timeout = 1 # The timeout in seconds for retriving the image associated to an URL
  batch_size = 1000 # The size of the batch for downloading images

  output_dataset_name = "improved_aesthetics_6.25plus_images_and_text"
  output_dir = "/home/yoann/Desktop/test_data"

# Create the configuration
config = DownloadConfig()

# Get the dataset to process
dataset: Dataset = load_dataset(config.dataset_to_download_name, split=f"train[:{config.num_entries}]")

# Get asynchronously the images for a list of URLs
async def get_images(urls: list):
    # Helper function to get an image from URL (or None if it cannot be retrieve)
    async def get_image(session: ClientSession, url: str) -> Image.Image or None:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        timeout = ClientTimeout(total=config.timeout)

        try:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    return None
                
                image_bytes = await response.read()
                raw_image = Image.open(BytesIO(image_bytes))
                return raw_image

        except Exception as e:
            return None

    # Try to get all the images
    async with aiohttp.ClientSession() as session:
        tasks = [get_image(session, url) for url in urls]
        images = await asyncio.gather(*tasks)
    return images


# Map each sample to their associated image and label
def map_samples(samples):
    async def async_helper(urls, texts):
        images = await get_images(urls)
        return {"image": images, "text": texts}
    
    return asyncio.run(async_helper(samples["URL"], samples["TEXT"]))


# Filter mapped samples where the image was not retrieved
def filter_func(sample):
  return sample["image"] != None

# The mapped dataset with images that were retrived linked to their label
mapped_dataset = dataset.map(map_samples, batched=True, batch_size=config.batch_size).filter(filter_func).select_columns(["image", "text"])

# Generator to create the final dataset
def gen_final_dataset(dataset: Dataset):
  def _gen():
    for sample in dataset:
      yield sample
  return _gen

# Create the final dataset with the new name and save it on the disk
final_dataset = Dataset.from_generator(generator=gen_final_dataset(mapped_dataset), dataset_name=config.output_dataset_name)
final_dataset.save_to_disk(config.output_dir)
