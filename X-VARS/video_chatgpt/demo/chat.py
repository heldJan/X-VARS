import time
import torch
import gradio as gr
from video_chatgpt.utils import (build_logger)
from conversation_discussion import conv_templates, SeparatorStyle
from conversation_discussion import load_video
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import logging
from video_chatgpt.constants import *
import numpy as np
from visual_encoder.config.classes import INVERSE_EVENT_DICTIONARY
import pickle
import json


logging.basicConfig(level=logging.WARNING)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Video-ChatGPT"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


class Chat:
    def __init__(self, model_name, conv_mode, tokenizer, image_processor, vision_tower, model, replace_token):
        self.model_name = model_name
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_tower = vision_tower
        self.model = model
        self.replace_token = replace_token

    def upload_video(self, video, img_list):
        if isinstance(video, str):  # is a path
            frames = load_video(video)

            start_frame = 63
            end_frame = 87
            fps = 17
            fps_beginning = 25
            factor = (end_frame - start_frame) / (((end_frame - start_frame) / fps_beginning) * fps)

            frames = frames[start_frame:end_frame]
            frames = self.image_processor.preprocess(frames, return_tensors='pt')['pixel_values']

            final_frames = None

            # We extract as many frames in order to have 17 frames per second
            for j in range(len(frames)):
                if j%factor<1:
                    if final_frames == None:
                        final_frames = frames[j,:,:,:].unsqueeze(0)
                    else:
                        final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

            img_list.append(final_frames.cuda())
        else:
            raise NotImplementedError
        msg = "Received."
        return msg
    

    def get_spatio_temporal_features_torch(self, features, num_temporal_tokens=44):
        t, s, c = features.shape

        temporal_tokens = np.mean(features, axis=1)
        padding_size = num_temporal_tokens - t
        if padding_size > 0:
            temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

        spatial_tokens = np.mean(features, axis=0)
        sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

        return sp_features

    def answer(self, state, img_list, temperature, max_new_tokens, first_run):
        if state.skip_next:
            # This generates call is skipped due to invalid inputs
            yield (state, state.to_gradio_chatbot(), img_list, first_run) + (no_change_btn,) * 5
            return
        
        image_tensor = img_list[0]
        # Generate video spatio-temporal features
        image_tensor = image_tensor.cuda()


        with torch.no_grad():
            out_off, out_act, frame_features = self.vision_tower(image_tensor)

            preds_sev = torch.argmax(out_off.detach().cpu(), 0)
            preds_act = torch.argmax(out_act.detach().cpu(), 0)

            values = {}
            action_class = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
            if preds_sev.item() == 0:
                offence_class = "No offence"
                severity_class = ""
            elif preds_sev.item() == 1:
                offence_class = "Offence"
                severity_class = "1.0"
            elif preds_sev.item() == 2:
                offence_class = "Offence"
                severity_class = "3.0"
            elif preds_sev.item() == 3:
                offence_class = "Offence"
                severity_class = "5.0"

            if offence_class == "Offence":
                offence_class = ", foul"

            if offence_class == "No offence":
                offence_class = " and no foul "

            if severity_class == "1.0":
                offence_class += " and no card"
            
            if severity_class == "3.0":
                offence_class += " and a yellow card"

            if severity_class == "5.0":
                offence_class += " and a red card"

            if action_class == "Tackling":
                action_class = "a tackle"
            if action_class == "Standing tackling":
                action_class = "a foot duel"
            if action_class == "Elbowing":
                action_class = "using his elbows or arms"
            if action_class == "Holding":
                action_class = "holding"
            if action_class == "High leg":
                action_class = "a high leg"
            if action_class == "Pushing":
                action_class = "pushing"
            if action_class == "Challenge":
                action_class = "a shoulder challenge"
            if action_class == "Dive":
                action_class = "a simulation"
            
        if first_run:
            conv_mode = self.conv_mode
            new_state = conv_templates[conv_mode].copy()
            new_state.append_message(new_state.roles[0], state.messages[-2][1])
            new_state.append_message(new_state.roles[1], None)
            state = new_state
            first_run = False


        f = open("annotations/predictionsTest_clip.json")
        data = json.load(f)
        pred = data["Actions"]

        path = "/gpfs/scratch/acad/telim/VARS/dataset/Test/action_115/PRE_CLIP_feature_clip_3.pkl"

        pred_action = pred[path]["Action class"]
        pred_off = pred[path]["Offence"]
        pred_card = pred[path]["Severity"]

        if pred_off == "Offence":
            pred_off = ", foul"

        if pred_off == "No offence":
            pred_off = " and no foul "

        if pred_card == "1.0":
            pred_off += " and no card"
        
        if pred_card == "3.0":
            pred_off += " and a yellow card"

        if pred_card == "5.0":
            pred_off += " and a red card"

        if pred_action == "Tackling":
            pred_action = "a tackle"
        if pred_action == "Standing tackling":
            pred_action = "a foot duel"
        if pred_action == "Elbowing":
            pred_action = "using his elbows or arms"
        if pred_action == "Holding":
            pred_action = "holding"
        if pred_action == "High leg":
            pred_action = "a high leg"
        if pred_action == "Pushing":
            pred_action = "pushing"
        if pred_action == "Challenge":
            pred_action = "a shoulder challenge"
        if pred_action == "Dive":
            pred_action = "a simulation"

        state.set_predictions(pred_action, pred_off)
        # Construct prompt
        prompt = state.get_prompt()

        prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, self.replace_token, 1)

        inputs = self.tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = state.sep if state.sep_style != SeparatorStyle.TWO else state.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Uncomment this for debugging purposes
        # pload = {
        #     "model": self.model_name,
        #     "prompt": prompt,
        #     "temperature": float(temperature),
        #     "max_new_tokens": min(int(max_new_tokens), 1536),
        #     "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        # }
        # logger.info(f"==== request ====\n{pload}")

        state.messages[-1][-1] = ""
        yield (state, state.to_gradio_chatbot(), img_list, first_run) + (disable_btn,) * 5

        video_spatio_temporal_features = self.get_spatio_temporal_features_torch(frame_features.numpy().astype("float16"))

        video_spatio_temporal_features = torch.from_numpy(video_spatio_temporal_features).unsqueeze(0)
        video_spatio_temporal_features = video_spatio_temporal_features.cuda()

        with open(path, 'rb') as f:
            video_spatio_temporal_features = torch.from_numpy(pickle.load(f)).cuda()
            video_spatio_temporal_features = video_spatio_temporal_features.unsqueeze(0)


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features,
                do_sample=False,
                temperature=float(temperature),
                max_new_tokens=min(int(max_new_tokens), 1536),
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        output = post_process_code(outputs)
        for character in output:
            state.messages[-1][-1] += character
            time.sleep(0.01)
            yield (state, state.to_gradio_chatbot(), img_list, first_run) + (enable_btn,) * 5
        # state.messages[-1][-1] = state.messages[-1][-1][:-1]
        logger.info(f"{output}")
        yield (state, state.to_gradio_chatbot(), img_list, first_run) + (enable_btn,) * 5
