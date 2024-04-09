import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from video_chatgpt.train.llava_trainer import VideoChatGPTTrainer
from video_chatgpt.model import *
import copy
import pathlib
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from video_chatgpt.train.llava_trainer import VideoChatGPTTrainer
from video_chatgpt import video_conversation as conversation_lib
from video_chatgpt.model import *
import torch.distributed as dist
from video_chatgpt.constants import *
import pickle
import json
import random
import transformers
from custom_dataset import VARS_Explain
import torch.distributed as dist
import gc
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria

from accelerate import Accelerator

# Base model
base_model = 'base_model_videoChatGPT'

#Where to save the new model
new_model = "Train_for_3_epoch_TEST"

device_index = Accelerator().process_index
device_map = {"": device_index}

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(
            'base_model_videoChatGPT',
            cache_dir=None,
            model_max_length=1048,
            max_length=64,
            padding_side="right",
            use_fast=False,
    )

# Add video token, start_video token and end_video token
tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

# Create the dataset
json_path_train = "annotations/SoccerNet-XFoul_train.json"
json_path_valid = "annotations/SoccerNet-XFoul_test.json"
json_path_train_predictions = "annotations/CLIP_prediction_train.json"
json_path_valid_predictions = "annotations/CLIP_prediction_test.json"

dataset_path = "/gpfs/scratch/acad/telim/VARS/dataset"
video_token_len = 300
test_dataset = VARS_Explain(json_path_valid, json_path_valid_predictions, dataset_path, video_token_len, tokenizer, 'Valid')
train_dataset = VARS_Explain(json_path_train, json_path_train_predictions, dataset_path, video_token_len, tokenizer, 'Train')

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration 
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['mm_projector', 'upsample_features', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)


# Load base moodel
# For now I load the model without quantization as I want to first figure out how to load the pretrained weights to the projection layer
model = VideoChatGPTLlamaForCausalLM.from_pretrained(
    base_model,
    mm_hidden_size=1024,
    quantization_config=bnb_config,
    cache_dir=None,
    device_map=device_map,
    #torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
)

model_vision_dict = model.get_model().initialize_vision_modules(
        pretrain_mm_mlp_adapter=None
    )
vision_config = model_vision_dict['vision_config']


# Predefine some constants
model.config.tune_mm_mlp_adapter = True
model.config.freeze_mm_mlp_adapter = False
model.config.mm_use_vid_start_end = True
mm_use_vid_start_end = True
vision_config.use_vid_start_end = True
model.config.sep_video_conv_front = False

model.initialize_vision_tokenizer(mm_use_vid_start_end=True, tokenizer=tokenizer,
                                      device='cuda', tune_mm_mlp_adapter=True,
                                      pretrain_mm_mlp_adapter='')


model.resize_token_embeddings(len(tokenizer))

vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
vision_config.use_vid_start_end = mm_use_vid_start_end
vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])


# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
model = prepare_model_for_kbit_training(model)

# Set training arguments
training_arguments = TrainingArguments(
        output_dir="results/",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_steps=1,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        lr_scheduler_type="constant",
        report_to="wandb",
        save_strategy="epoch",
        num_train_epochs = 3, # Number of steps
        ddp_find_unused_parameters=False
)

# Set supervised fine-tuning parameters
trainer = VideoChatGPTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="question",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Reload model in FP16 and merge it with LoRA weights
model = VideoChatGPTLlamaForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained('LLaVA-7B-Lightening-v1-1', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


