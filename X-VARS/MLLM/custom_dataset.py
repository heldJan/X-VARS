
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
from video_chatgpt.model import *
import copy
import pathlib
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle
import json
import random
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle


# Predefine some constants and templates

IGNORE_INDEX = -100
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

#template_text = "You are an artificial intelligence assistant for visual football referee questions. Analyze the following football/soccer video clip carefully and answer the asked question. Provide explanations based on the laws of football to justify your decision. Give helpful answers to the user's questions. USER: "
template_text = "You are an artificial intelligence assistant for visual football referee questions. Give short and helpful answers to the user's questions. USER: "


answer_id = "ASSISTANT: "

class VARS_Explain(torch.utils.data.Dataset):
    def __init__(self, json_path, json_path_predictions, dataset, video_token_len, tokenizer, split):

        self.path_dataset = dataset
        self.tokenizer = tokenizer
        self.video_token_len = video_token_len

        f_gt = open("annotations/annotations_GT.json")
        self.data_gt = json.load(f_gt)
        
        # f contains the questions and answers
        f = open(json_path)
        self.data = json.load(f)
        self.conv_mode = "video-chatgpt_v1"
        self.split = split

        # f2 contains the prediction of our MViT classifier pretrained on classifying if it is a foul or not, the severity and the type of foul.
        # Could be interesting to input the prediction of the classifier along with the video tokens obtained from CLIP
        f2 = open(json_path_predictions)
        self.pred = json.load(f2)
        self.pred = self.pred["Actions"]


    def preprocess(self, idx: int, path_prediction: str, short_path_prediction: str) -> Dict:

        sep2 = "</s>"

        # Load question and answer
        question = self.data[idx]["question"]
        answer = self.data[idx]["answer"]

        # We check what prediction the Fine-tuned classifier has made for this action
        pred_action = self.pred[path_prediction]["Action class"]
        pred_off = self.pred[path_prediction]["Offence"]
        pred_card = self.pred[path_prediction]["Severity"]

        if pred_off == "Offence":
            pred_off = ", foul and "

        if pred_off == "No offence":
            pred_off = "and no foul."

        if pred_card == "1.0":
            pred_off += "no card."
        
        if pred_card == "3.0":
            pred_off += "a yellow card."

        if pred_card == "5.0":
            pred_off += "a red card."

        if pred_action == "Tackling":
            pred_action = "a tackle "
        if pred_action == "Standing tackling":
            pred_action = "a foot duel "
        if pred_action == "Elbowing":
            pred_action = "using his elbows or arms "
        if pred_action == "Holding":
            pred_action = "holding "
        if pred_action == "High leg":
            pred_action = "a high leg "
        if pred_action == "Pushing":
            pred_action = "pushing "
        if pred_action == "Challenge":
            pred_action = "a shoulder challenge "
        if pred_action == "Dive":
            pred_action = "a simulation "

        qs = question + " The prediction for this video is " + pred_action + pred_off + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len + DEFAULT_VID_END_TOKEN
        
        conv = conv_templates[self.conv_mode].copy()
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        # Until now no conversation but only question and answer
        conversations = []
        conversations.append(prompt)

        # Tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        targets = input_ids.clone()

        # Mask targets
        # We mask everything except from the answer
        sep = "ASSISTANT:"
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
            rounds = conversation.split(sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return dict(
            input_ids=input_ids.squeeze(),
            labels=targets.squeeze(),
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).squeeze(),
        )

    def __getitem__(self, idx):

        ################## LOADING FEATURES ###############
        # Single View
        # For the same action, we have sometimes 1 replay, sometimes 2 and sometimes 3
        # Here we randomly pick one of the views
        counter = 1
        if os.path.exists(self.path_dataset + "/" + self.data[idx]["path"] + "/PRE_CLIP_feature_clip_2.pkl"):
            counter += 1
        if os.path.exists(self.path_dataset + "/" + self.data[idx]["path"] + "/PRE_CLIP_feature_clip_3.pkl"):
            counter += 1

        # We randomly load the pre-calculated features from CLIP
        i = random.randint(1, counter)
        with open(self.path_dataset + "/" + self.data[idx]["path"] + "/PRE_CLIP_feature_clip_" + str(i) + ".pkl", 'rb') as f:
            video_spatio_temporal_features = torch.from_numpy(pickle.load(f))

        i = 1
        item = self.preprocess(idx, self.path_dataset + "/" + self.data[idx]["path"] + "/PRE_CLIP_feature_clip_" + str(i) + ".pkl", self.data[idx]["path"])
        item["video_spatio_temporal_features"] = video_spatio_temporal_features

        return item

    def __len__(self):
        return len(self.data)
