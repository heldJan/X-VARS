"""
How to run this file:

cd VideoChatGPT
python -m video_chatgpt.single_video_inference \
    --model-name <path of llava weights, for eg "LLaVA-7B-Lightening-v1-1"> \
    --projection_path <path of projection for eg "video-chatgpt-weights/video_chatgpt-7B.bin"> \
    --video_path <video_path>
"""

from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch

#add new packages as below
from PIL import Image
from decord import VideoReader, cpu
from video_chatgpt.eval.model_utils import initialize_model, load_video
import argparse
import numpy as np
import os
import pickle
import timm
# Define constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_END_TOKEN = "\n"
DEFAULT_VIDEO_TOKEN = "<video>"

DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

template_text = "You are an artificial intelligence assistant for visual football referee questions. Give short and helpful answers to the user's questions. USER: "
user_id = " USER: "
answer_id = "ASSISTANT: "


def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    pred_action = "Tackling"
    pred_off = "Offence + Yellow card"

    #prompt = template_text + question + "Classifier prediction for type of foul: " + pred_action + ", Classifier prediction for offence and severity: " + pred_off + "\n" + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * 300 + DEFAULT_VID_END_TOKEN + answer_id 
    prompt = template_text + question + "Classifier prediction for type of foul: " + pred_action + ", Classifier prediction for offence and severity: " + pred_off + "\n" + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * 300 + DEFAULT_VID_END_TOKEN + answer_id 
    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    """image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']


    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().cuda()"""

    # Generate video spatio-temporal features
    """with torch.no_grad():
        print(image_tensor.shape)
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA

        print(frame_features.shape)"""

    action = "2"
    set_l = "Test"

    path_to_dataset = "/gpfs/scratch/acad/telim/VARS/dataset/"
    
    with open(path_to_dataset + set_l + "/action_" + action + "/PRE_CLIP_feature_clip_3.pkl", 'rb') as f:
        video_spatio_temporal_features = torch.from_numpy(pickle.load(f)).cuda()

    print(video_spatio_temporal_features.shape)

    # CLIP
    """with open("/gpfs/scratch/acad/telim/VARS/dataset/" + set_l + "/action_" + action + "/feature_clip_0.pkl", 'rb') as f:
        print("1")
        features = torch.from_numpy(pickle.load(f)).unsqueeze(0).cuda()

    with open("/gpfs/scratch/acad/telim/VARS/dataset/" + set_l + "/action_" + action + "/feature_clip_1.pkl", 'rb') as f:
        #video_spatio_temporal_features = torch.from_numpy(pickle.load(f)).cuda()

        features = torch.cat((features, torch.from_numpy(pickle.load(f)).unsqueeze(0).cuda()), dim=0)

    if os.path.exists("/gpfs/scratch/acad/telim/VARS/dataset/" + set_l + "/action_" + action + "/feature_clip_2.pkl"):
        print("3")
        with open("/gpfs/scratch/acad/telim/VARS/dataset/" + set_l + "/action_" + action + "/feature_clip_2.pkl", 'rb') as f:
            features = torch.cat((features, torch.from_numpy(pickle.load(f)).unsqueeze(0).cuda()), dim=0)

    if os.path.exists("/gpfs/scratch/acad/telim/VARS/dataset/" + set_l + "/action_" + action + "/feature_clip_3.pkl"):
        print("4")
        with open("/gpfs/scratch/acad/telim/VARS/dataset/" + set_l + "/action_" + action + "/feature_clip_3.pkl", 'rb') as f:
            features = torch.cat((features, torch.from_numpy(pickle.load(f)).unsqueeze(0).cuda()), dim=0)

    video_spatio_temporal_features = torch.max(features, dim=0)[0].squeeze()"""


    #video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)


    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        print("START")
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])
        print("DONE")

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')


    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-name", type=str, required=False)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False, default="")
    parser.add_argument("--video_path", type=str, required=False, default="")
    parser.add_argument("--conv_mode", type=str, required=False, default='video-chatgpt_v1')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # export PYTHONPATH="./:$PYTHONPATH"

    args = parse_args()


    """model_reloaded = timm.create_model('hf_hub:heldJan/llama-2-7b-froozen_mvit_test', pretrained=True)
    model_reloaded.cuda()

    print("NICE")"""

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model('pretrained_video_chatgpt', '')

    """video_path = "clip_1.mp4"

    if os.path.exists(video_path):
        video_frames = load_video(video_path)"""
    
    question = "What card would you give? Why?"
    conv_mode = args.conv_mode


    # Run inference on the video and add the output to the list
    output = video_chatgpt_infer("video_frames", question, conv_mode, model, vision_tower,
                                            tokenizer, image_processor, video_token_len)
    print("\n\n", output)
        