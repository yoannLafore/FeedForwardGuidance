# Feed-Forward Guidance for Text-to-Image Diffusion Models

Guidance mechanisms such as Classifier-Free Guidance have achieved remarkable results in improving the generation quality of Latent Diffusion Models (LDMs). However, these techniques typically require additional forward/backward passes during inference, which slows down the generation process. We propose distilling the guidance during training to retain its benefits without the inference overhead.

This repository has been created as part of the Bachelor Semester Research Project in Computer Science at EPFL.

A detailed report on this project, including the methodology, experiments, and results, is available [here](docs/FeedForwardGuidance_Report.pdf).

## Description

This repository provides the implementation to distill and evaluate Classifier-Free Guidance, CLIP guidance, as well as a blue tone (toy example) guidance. The code is organized into three main categories as follows:

For each category, a sample shell launch script is provided as an example.

### Pre-processing
This section is located under the `/dataset_preprocessing` directory and is divided into three sub-categories, each corresponding to a different guidance technique.\
Its purpose is to take an image dataset in HuggingFace Dataset format and generate a new dataset that can be used to distill the corresponding guidance into a stable diffusion model.

The `preprocess.py` script can be launched with various arguments depending on your needs. These arguments are detailed at the beginning of the script.

### Fine-tuning
This section is located under the `/training` directory and is also divided into three sub-directories (one per guidance technique).\
The goal is to distill the corresponding guidance technique into a stable diffusion model using the dataset generated in the previous step. The script used for this is `finetuning.py`. Once again, parameters for the training can be found at the beginning of the script.\
**Note:** The model being trained must be the same as the one used to generate the dataset.

### Evaluation
This section is located under the `/evaluation` directory and is divided into three sub-directories (one per guidance technique).\
It aims to evaluate the CLIP and FID scores of the CLIP-guided, Classifier-Free guided, and fine-tuned models.\
The scripts can be found under `clip_score.py` and `fid_score.py`.\
**Note:** To compute the FID/CLIP score of the fine-tuned model, you can use the evaluation script for the Classifier-Free guidance with a guidance scale of 1 (which effectively means no guidance).

## Acknowledgments

I would like to thank [Martin Nicolas Everaert](https://martin-ev.github.io/), my project supervisor at EPFL, for his invaluable guidance and support throughout this research project.

## References

This repository heavily relies on the Diffusers library, which can be found here: [Hugging Face Diffusers](https://github.com/huggingface/diffusers).
