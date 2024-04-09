import os
from visual_encoder.model import MVNetwork
import torch
from torchvision.io.video import read_video
from torchvision.models.video import MViT_V2_S_Weights
import pickle
import math
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from visual_encoder.config.classes import INVERSE_EVENT_DICTIONARY
from decord import VideoReader, cpu
import json
import shutil

def load_video(vis_path, num_frm=1000):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=44):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)

    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)

    print(spatial_tokens.shape)

    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features

def get_features(video_path):
    # Foul is mostly at the 3 second (at the 75th frame)
    start_frame = 63
    end_frame = 87
    fps = 17
    fps_beginning = 25
    factor = (end_frame - start_frame) / (((end_frame - start_frame) / fps_beginning) * fps)
    video_frames = load_video(video_path)
    
    # We only take the frames between 55 and 95 and not the whole video
    video_frames = video_frames[start_frame:end_frame]

    frames = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    
    final_frames = None

    # We extract as many frames in order to have 17 frames per second
    for j in range(len(frames)):
        if j%factor<1:
            if final_frames == None:
                final_frames = frames[j,:,:,:].unsqueeze(0)
            else:
                final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

    final_frames = final_frames.cuda()

    print(final_frames.shape)
    
    out_off, out_act, video_features = model(final_frames)

    print(video_features.shape)


    preds_sev = torch.argmax(out_off.detach().cpu(), 0)
    preds_act = torch.argmax(out_act.detach().cpu(), 0)

    values = {}
    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
    if preds_sev.item() == 0:
        values["Offence"] = "No offence"
        values["Severity"] = ""
    elif preds_sev.item() == 1:
        values["Offence"] = "Offence"
        values["Severity"] = "1.0"
    elif preds_sev.item() == 2:
        values["Offence"] = "Offence"
        values["Severity"] = "3.0"
    elif preds_sev.item() == 3:
        values["Offence"] = "Offence"
        values["Severity"] = "5.0"

    video_clip_features = get_spatio_temporal_features(video_features.numpy().astype("float16"))

    exit()

    return video_clip_features, values

# ENTER PATH TO DATASET
path_dataset = "/gpfs/scratch/acad/telim/VARS/dataset"
splits_list = ["Train", "Valid", "Test"]
image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
"""vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()"""


model = MVNetwork().cuda()
#model = MVNetwork(net_name=pre_model, agr_type=pooling_type).cuda()

path_weights = "visual_encoder/14_model.pth.tar"
load = torch.load(path_weights)
model.load_state_dict(load['state_dict'])

model.eval()

for split in splits_list:
    path_dataset_split = path_dataset + "/" + split
    folders = 0

    print(path_dataset_split)

    for _, dirnames, _ in os.walk(path_dataset_split):
        folders += len(dirnames) 


    print(folders)

    data = {}
    data["Set"] = split
    actions = {}
    
    for i in range(folders):
        path_clip = path_dataset_split + "/action_" + str(i)

        print(path_clip)

        if os.path.exists(path_clip + "/clip_1.mp4"):  
            features, values = get_features(path_clip + "/clip_1.mp4")

            """with open(path_clip + "/PRE_CLIP_feature_clip_1.pkl", 'wb') as f:
                pickle.dump(features, f)"""

            actions[path_clip + "/PRE_CLIP_feature_clip_1.pkl"] = values

        if os.path.exists(path_clip + "/clip_2.mp4"):  
            features, values = get_features(path_clip + "/clip_2.mp4")

            """with open(path_clip + "/PRE_CLIP_feature_clip_2.pkl", 'wb') as f:
                pickle.dump(features, f)"""
            
            actions[path_clip + "/PRE_CLIP_feature_clip_2.pkl"] = values

        if os.path.exists(path_clip + "/clip_3.mp4"):  
            features, values = get_features(path_clip + "/clip_3.mp4")

            """with open(path_clip + "/PRE_CLIP_feature_clip_3.pkl", 'wb') as f:
                pickle.dump(features, f)"""
            
            actions[path_clip + "/PRE_CLIP_feature_clip_3.pkl"] = values

    data["Actions"] = actions
    with open("predictions" + split + "_clip.json", "w") as outfile: 
        outfile.write(json.dumps(data, indent=4))

            
