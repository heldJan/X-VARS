from video_chatgpt.model import video_chatgpt
import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.constants import *
import torch
from peft import PeftModel


model_name = 'LLaVA-7B-Lightening-v1-1'
projection_path = 'video_chatgpt-7B.bin'

# Load model
model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                         use_cache=True, device_map='cpu')
# Load the weights from projection_path after resizing the token_embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add tokens to tokenizer
tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)

tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

# Resize token embeddings of the model
model.resize_token_embeddings(len(tokenizer))

if projection_path:
    status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
    if status.unexpected_keys:
        print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
    print(f"Weights loaded from {projection_path}")

model.save_pretrained("base_model_videoChatGPT")
tokenizer.save_pretrained("base_model_videoChatGPT")

"""model_name = 'pretrained_video_chatgpt'

# Load model
model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                         use_cache=True, device_map='cpu')
# Load the weights from projection_path after resizing the token_embeddings
tokenizer = AutoTokenizer.from_pretrained('LLaVA-7B-Lightening-v1-1')

# Add tokens to tokenizer
tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)

tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

# Resize token embeddings of the model
model.resize_token_embeddings(len(tokenizer))

path_to_llama_model = 'Train_for_3_epoch_TEST'

model = PeftModel.from_pretrained(model, path_to_llama_model)
model = model.merge_and_unload()

model.save_pretrained("MODEL_Train_for_3_epoch_Test")"""
