# Training

This README outlines the steps required for training. We train X-VARS on 22k video-question-answer triplets. The training on 2 A100 40GB GPUs takes about 2 hours.

## Download weights
Download the [base_model_videoChatGPT](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing) weights of our [Video-ChatGPT](https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing) language model backbone.
Save the weights in "X-VARS/X-VARS".

## Download the SoccerNet-XFoul dataset

The annotations will be available soon! Stay tuned🔥

## Train X-VARS
To run your code on multiple GPUs, run the following:

```
accelerate launch --config_file "path/to/default_config.yaml" training.py
```

All information needed to create the "default_congif.yaml" file are provided by [HuggingFace](https://huggingface.co/docs/accelerate/en/package_reference/cli).
