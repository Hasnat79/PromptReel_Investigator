"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
import json
from tqdm import tqdm


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
 
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
print(args.gpu_id)

model_cls = registry.get_model_class(model_config.arch)

model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')



def reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return  chat_state, img_list
#-----------------------------------------------
def upload_video(gr_video, chat_state,audio_flag):
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    
    if gr_video is not None:
        print(gr_video)
        chat_state.system =  ""
        img_list = []
        if audio_flag:
            llm_message = chat.upload_video(gr_video, chat_state, img_list)
        else:
            llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return  chat_state, img_list
    #---------------------------------------------


#-------------------------------
def gradio_ask(user_message,chat_state):
    if len(user_message) == 0:
        return 'Input should not be empty!'
    chat.ask(user_message, chat_state)
    return chat_state
#--------------------------------

def gradio_answer( chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    # print(f"chat_state.get_prompt(): {chat_state.get_prompt()}")
    # print(f"chat_state: {chat_state}")
    # print(f"llm message: {llm_message}")
    return chat_state, img_list, llm_message

def write_json(output_file_name,data):
    with open(output_file_name,'w') as f:
        json.dump(data,f,indent =4)

def open_json(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
    return data
#TODO show examples below


def videollama_output_generation(video_path,text_input):
    num_beams = 1
    temperature = 1

    audio_flag = 1
    chat_state = default_conversation.copy()
    # def upload_video(gr_video, chat_state,audio_flag):
    chat_state,image_list = upload_video(video_path,chat_state,audio_flag)
    # asking question to llm
    chat_state = gradio_ask (text_input, chat_state)
    # extracting asnwer from llm
    chat_state,image_list, llm_message = gradio_answer(chat_state,image_list,num_beams,temperature)

    # print(f"llm message: {llm_message}")

    chat_state, image_list = reset(chat_state,image_list)
    return llm_message

def select_random_videos(directory, exclude_file, num_videos):
    # Get a list of all mp4 files in the directory
    all_videos = [file for file in os.listdir(directory) if file.endswith(".mp4")]

    # Remove the excluded file from the list
    all_videos = [video for video in all_videos if video != exclude_file]

    # Select num_videos random videos
    selected_videos = random.sample(all_videos, min(num_videos, len(all_videos)))

    return selected_videos



def main():
    oops_dataset_dir = "PromptReel_Investigator/oops_all_failed_videos_val_dir"

    # excluding this file from the selection of videos, because we are using the text prompt that is made from this video (randomly selected this video)
    exclude_file = "34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)5.mp4"
    
    
    exp_data_file_name = "PromptReel_Investigator/results/__irr_vid_prmt_exp_data.json"
    exp_data = []

    text_input = "Question: usually people would not do this activity, that is surprising. consider the following events in the video:'0:00-0:02 both kids are eating their breakfast', '0:03-0:04 Kid wearing black shirt suddenly falls', '0:07-0:10 kid with red shirt falls too', '0:11-0:12 kid wearing red shirt starts crying'"

    #randomly selected 100 videos from the oops validation set
    num_videos_to_select = 100
    selected_videos = select_random_videos(oops_dataset_dir,exclude_file,num_videos_to_select)
    with tqdm(total = 100) as pbar:
        for video in selected_videos:
            video_path = os.path.join(oops_dataset_dir,video)
            llm_message = videollama_output_generation(video_path,text_input)
            res = {
                'video_path': video_path,
                'videollama_ouput':llm_message
            }
            exp_data.append(res)
            pbar.update(1)
        
    write_json(exp_data_file_name,exp_data)


if __name__ == "__main__":
    main()
# %%



