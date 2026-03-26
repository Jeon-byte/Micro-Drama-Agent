import os
import pdb
import json
import torch
import numpy as np
from os import path
from pathlib import Path
# import sys
# sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))

from .inference_with_id import process_generation, array2string


def save_results(save_folder, result, img_name):
    folder_name = save_folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for idx, img in enumerate(result):
        if idx == 0: continue
        file_path = os.path.join(folder_name, f"{img_name}")
        img.save(file_path)
    
def construct_args_run(save_folder, img_name, upload_images, general_prompt, prompt_array):
    sd_type, model_type = "Unstable", "Using Ref Images"
    sa32_, sa64_ = 0.5, 0.5 # network
    num_steps, guidance_scale = 35, 5.0 # diffusion
    Ip_Adapter_Strength, style_strength_ratio = 0.5, 20 # module weight
    
    seed = 0
    id_length = 1
    ## prompt for the characters
    prompt_array = array2string(prompt_array)

    # print("----------------")
    # print(prompt_array)
    negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs"
    style_name, style = 'Photographic', 'Photographic' # image style
    comic_type = ''
    G_height, G_width = 864, 1536
    font_choice = 'Inkfree.ttf'
    char_path = ''
    args_list = [sd_type, model_type, upload_images, num_steps, style_name,
                Ip_Adapter_Strength, style_strength_ratio, guidance_scale,
                seed, sa32_, sa64_, id_length,
                general_prompt, negative_prompt, prompt_array,
                G_height, G_width,
                comic_type,
                font_choice, char_path]
    result = process_generation(*args_list)

    save_results(save_folder, result, img_name)
    
def inference(data):
    story_root, img_root, save_root = data[0], data[1], data[2]
    ## load the stories
    story_dic = {}
    if type(story_root) is str:
        for file_name in os.listdir(story_root):
            with open(path.join(story_root, file_name), 'r') as f:
                story_dic[file_name] = json.load(f)
    elif type(story_root) is list:
        for file_name in story_root:
            with open(file_name, 'r') as f:
                story_dic[file_name.split('/')[-1]] = json.load(f)
                
    ## load the image_path
    img_dic = {}
    for img_name in os.listdir(img_root):
        char_name = img_name.split('-')[-1].replace('_', '').replace(' ', '') # get char name
        img_dic[char_name] = path.join(img_root, img_name, 'best.jpg')
    ## align character with stories
    char_story_dic = {}
    char_story_dic['NC'] = []
    for file_name in story_dic.keys():
        story = story_dic[file_name]
        char_list = story['Characters']
        if len(char_list) == 0: # no character, belong to NC
            char_story_dic['NC'].append([file_name, story])
        else:
            save_flag = False
            for char in char_list:
                char = char.replace('_', '').replace(' ', '')
                if char in img_dic.keys():
                    if char not in char_story_dic.keys(): 
                        char_story_dic[char] = []
                    char_story_dic[char].append([file_name, story])
                    save_flag = True
                    break # only control one character
            if save_flag is False: # no character img, belong to NC
                char_story_dic['NC'].append([file_name, story])
    ## inference for each character
    prompt_array_nc = []
    img_name_nc = []
    for story in char_story_dic['NC']:
        story_line = '[NC] '+story[1]['Plot']
        prompt_array_nc.append(story_line)
        img_name_nc.append(story[0].replace('.json', ''))
        
    flag = False
    for char_name in char_story_dic.keys():
        if char_name == 'NC':
            continue
        general_prompt = '[{}], a {} img'.format(char_name, 'human')
        story_list = char_story_dic[char_name]    
        img_name = []
        prompt_array = []
        upload_images = [img_dic[char_name]]
        if len(story_list) == 1:
            story_list.extend(story_list)
        for story in story_list:
            story_line = '[{}] '.format(char_name)+limit_word_count(story[1]['Plot'])
            prompt_array.append(story_line)
            img_name.append(story[0].replace('.json', ''))
        if flag is False:
            prompt_array.extend(prompt_array_nc)
            img_name.extend(img_name_nc)
            flag = True
        
        # print("----------------")
        # print(save_root)
        # print(img_name)
        # print(upload_images)
        # print(general_prompt)
        # print(prompt_array)
        # print("----------------")

        construct_args_run(save_root, img_name, upload_images, general_prompt, prompt_array)

def demo_inference(data):
    story_root, img_root, save_root = data[0], data[1], data[2]
    ## load the stories
    story_dic = {}
    if type(story_root) is str:
        for file_name in os.listdir(story_root):
            with open(path.join(story_root, file_name), 'r') as f:
                story_dic[file_name] = json.load(f)
    elif type(story_root) is list:
        for file_name in story_root:
            with open(file_name, 'r') as f:
                story_dic[file_name.split('/')[-1]] = json.load(f)
                
    ## load the image_path
    img_dic = {}
    for img_name in os.listdir(img_root):
        char_name = img_name.split('-')[-1].replace('_', '').replace(' ', '') # get char name
        img_dic[char_name] = path.join(img_root, img_name, 'best.jpg')
    ## align character with stories
    char_story_dic = {}
    char_story_dic['NC'] = []
    for file_name in story_dic.keys():
        story = story_dic[file_name]
        char_list = story['Characters']
        if len(char_list) == 0: # no character, belong to NC
            char_story_dic['NC'].append([file_name, story])
        else:
            save_flag = False
            for char in char_list:
                char = char.replace('_', '').replace(' ', '')
                if char in img_dic.keys():
                    if char not in char_story_dic.keys(): 
                        char_story_dic[char] = []
                    char_story_dic[char].append([file_name, story])
                    save_flag = True
                    break # only control one character
            if save_flag is False: # no character img, belong to NC
                char_story_dic['NC'].append([file_name, story])
    ## inference for each character
    prompt_array_nc = []
    img_name_nc = []
    for story in char_story_dic['NC']:
        story_line = '[NC] '+story[1]['Plot']
        prompt_array_nc.append(story_line)
        img_name_nc.append(story[0].replace('.json', ''))
    
    general_prompt = ''
    img_name = []
    prompt_array = []
    upload_images = []
    flag = False
    for char_name in char_story_dic.keys():
        if char_name == 'NC':
            continue
        if len(general_prompt) == 0: general_prompt += '[{}], a {} img'.format(char_name, 'human')
        else: general_prompt += '\n[{}], a {} img'.format(char_name, 'human')

        story_list = char_story_dic[char_name]    
        upload_images.append(img_dic[char_name])
        
        while len(story_list) < 3:
            story_list.extend(story_list)
            
        for story in story_list:
            story_line = '[{}] '.format(char_name)+limit_word_count(story[1]['Plot'])+limit_word_count(story[1]['Background Description'])
            img_name.append(story[0].replace('.json', ''))
            prompt_array.append(story_line)
            
        if flag is False:
            prompt_array.extend(prompt_array_nc)
            img_name.extend(img_name_nc)
            flag = True
            
    construct_args_run(save_root, img_name, upload_images, general_prompt, prompt_array)
        
def limit_word_count(text, max_words=20):
    words = text.split()  
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def find_true_intervals(boolean_list):
    intervals = []
    start = None
    for i, value in enumerate(boolean_list):
        if value and start is None:
            start = i
        elif not value and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(boolean_list) - 1))
    return intervals

if __name__ == '__main__':
    save_root_path = '/users/zeyuzhu/movie_gen/StoryDiffusion/with_bg_result'
    story_root_path = '/storage/wuweijia/MovieGen/lsmdc/Annotation_Shot_Desc/Test'
    identity_root_path = '/storage/wuweijia/MovieGen/lsmdc/GT/Character_Bank/Char_Bank/Test'
    test_data_list = ['1004_Juno', '1017_Bad_Santa', '1027_Les_Miserables', '1040_The_Ugly_Truth', 
                      '1041_This_is_40', '1054_Harry_Potter_and_the_prisoner_of_azkaban']
    
    # print("----------------")
    save_root = "./test"
    img_name= "test"
    upload_images = ['/storage/wuweijia/MovieGen/lsmdc/GT/Character_Bank/Char_Bank/Test/1004_Juno/Jason_Bateman-Mark_Loring/best.jpg']
    general_prompt = "[MarkLoring], a human img"
    prompt_array = ['[MarkLoring] Mark opens the front door and catches sight of Juno driving off, indicating a moment of departure or farewell.']


    construct_args_run(save_root, img_name, upload_images, general_prompt, prompt_array)
    
    # for data_index in range(6):
    #     story_path = path.join(story_root_path, test_data_list[data_index])
    #     if_less_than_one_person = []
    #     for file_name in os.listdir(story_path):
    #         with open(path.join(story_path, file_name), 'r') as f: data = json.load(f)
    #         char_list = data['Characters']
    #         if len(char_list) <= 1: if_less_than_one_person.append(True)
    #         else: if_less_than_one_person.append(False)
    #     result = find_true_intervals(if_less_than_one_person)
    #     for period in result:
    #         story_path_list = []
    #         if period[1]-period[0] >= 10:
    #             for index in range(period[0], period[1]):
    #                 story_path_list.append(path.join(story_path, os.listdir(story_path)[index]))
    #             demo_inference([story_path_list,
    #             path.join(identity_root_path, test_data_list[data_index]),
    #             path.join(save_root_path, test_data_list[data_index]+'_{}-{}'.format(str(period[0]), str(period[1])))])
    
    # for data_index in range(6):
    #     inference([path.join(story_root_path, test_data_list[data_index]),
    #                path.join(identity_root_path, test_data_list[data_index]),
    #                path.join(save_root_path, test_data_list[data_index])])
    
# 1004 '[Rollo] not have enough prompt description, need no less than 2, but you give 1'
# 1017/ 1027'[PhotoElf] not have enough prompt description, need no less than 2, but you give 1'
# 1041 '[Georgia] not have enough prompt description, need no less than 2, but you give 1'
# 
