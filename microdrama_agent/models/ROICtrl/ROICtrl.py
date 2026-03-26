from PIL import Image, ImageDraw, ImageFont
from .roictrl.pipelines.pipeline_stable_diffusion_edlora import StableDiffusionEDLoRAPipeline, bind_concept_prompt
from .roictrl.models.unet_2d_condition_model import UNet2DConditionModel
import torch
import math
from copy import deepcopy
from safetensors.torch import load_file, save_file
import os
import json
from einops import rearrange
import numpy as np
import cv2
import random

# define the image path
def setup_pipeline(pretrained_model_path, roictrl_path, device='cuda'):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, attention_type='roictrl', subfolder="unet", torch_dtype=torch.float16, low_cpu_mem_usage=False, device_map=None)

    pretrained_state_dict = load_file(roictrl_path)
    unet.load_state_dict(pretrained_state_dict, strict=False)

    pipe = StableDiffusionEDLoRAPipeline.from_pretrained(
        pretrained_model_path,
        unet=unet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)

    return pipe

def draw_box(image, box_list, instance_list, height=320, width=512):
    image = deepcopy(image)
    draw = ImageDraw.Draw(image)

    for box, inst_caption in zip(box_list, instance_list):
        anno_box = deepcopy(box)
        xmin, xmax = anno_box[0] * width, anno_box[2] * width
        ymin, ymax = anno_box[1] * height, anno_box[3] * height

        draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=1)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        font = ImageFont.truetype('Rainbow-Party-2.ttf', 16)

        text_bbox = draw.textbbox((0, 0), f"{inst_caption}", font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.text((center_x - text_width / 2, center_y - text_height / 2), f"{inst_caption}", fill='red', font=font)

    return image

@torch.no_grad()
def encode_roi_input(input_data, pipe, device='cuda', max_rois=30, do_classifier_free_guidance=True, use_instance_cfg=True, negative_prompt=None):
    roi_boxes = input_data["roi_boxes"]
    roi_phrases = input_data["roi_phrases"]

    if len(roi_boxes) > max_rois:
        print(f"use first {max_rois} rois")
        roi_boxes = roi_boxes[: max_rois]
        roi_phrases = roi_phrases[: max_rois]
    assert len(roi_boxes) == len(roi_phrases), 'the roi phrase and position not equal'

    # encode roi prompt
    _roi_phrases = bind_concept_prompt(roi_phrases, pipe.new_concept_cfg)

    tokenizer_inputs = pipe.tokenizer(_roi_phrases, padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt").to(device)
    _instance_embedding = pipe.text_encoder(tokenizer_inputs.input_ids.to(device))[0]
    _instance_embedding = rearrange(_instance_embedding, "(b n) l d -> b n l d", n=16)

    if negative_prompt is None:
        uncond_text = ""
    else:
        uncond_text = negative_prompt
    uncond_tokenizer_inputs = pipe.tokenizer(uncond_text, padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt").to(device)
    uncond_text_res = pipe.text_encoder(uncond_tokenizer_inputs.input_ids.to(device))
    _instance_uncond_embedding = uncond_text_res[0]

    instance_boxes = torch.zeros(max_rois, 4, device=device, dtype=pipe.text_encoder.dtype)
    instance_embeddings = torch.zeros(
        max_rois, 16, pipe.tokenizer.model_max_length, pipe.unet.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
    )
    instance_masks = torch.zeros(max_rois, device=device, dtype=pipe.text_encoder.dtype)
    uncond_instance_embeddings = torch.zeros(
        max_rois, 16, pipe.tokenizer.model_max_length, pipe.unet.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
    )

    n_rois = len(roi_boxes)
    instance_boxes[:n_rois] = torch.tensor(roi_boxes)
    instance_embeddings[:n_rois] = _instance_embedding
    uncond_instance_embeddings[:n_rois] = _instance_uncond_embedding
    instance_masks[:n_rois] = 1

    if do_classifier_free_guidance:
        instance_boxes = torch.stack([instance_boxes] * 2)
        instance_embeddings = torch.stack([uncond_instance_embeddings, instance_embeddings])
        instance_masks = torch.stack([instance_masks] * 2)

        instance_boxemb_masks = instance_masks.clone()
        instance_boxemb_masks[0] = 0

        if not use_instance_cfg:
            instance_masks[0] = 0

    roictrl = {
        'instance_boxes': instance_boxes,
        'instance_embeddings': instance_embeddings,
        'instance_masks': instance_masks,
        'instance_boxemb_masks': instance_boxemb_masks
    }
    return roictrl


class ROICtrl_pipe:
    def __init__(self, pretrained_roictrl="", roictrl_path="", dtype = "bfloat16",device="cuda"):

        self.pipe = setup_pipeline(
            pretrained_model_path=pretrained_roictrl,
            roictrl_path=roictrl_path
        )
        self.vis = True


    def predict(self, prompt, refer_images, character_box, save_name, size, seed = 0):

        input_data = {
            "caption": "{}, 4K, high quality, high resolution, best quality".format(prompt),
            "roi_boxes": [
            ],
            "roi_phrases": [
            ],
            "height": size[1],
            "width": size[0],
            "seed": 42,
            "roictrl_scheduled_sampling_beta": 1.0
        }
        if character_box == {}:
            character_box["none"] = [0,0,0.001,0.001]
            
        for name in character_box:
            y1,y2 = random.uniform(0.00001, 0.2),random.uniform(0.8, 0.9999)
            character_box[name][1], character_box[name][3] = y1,y2
            box = character_box[name]
            
            strr = "a <{}1> <{}2>".format(name,name)

            input_data["roi_boxes"].append(box)
            input_data["roi_phrases"].append(strr)
            # input_data["caption"] = input_data["caption"].replace(name,"a people")

        print("-----  ------")
        print(input_data)

        cross_attention_kwargs = {
            'roictrl': encode_roi_input(input_data, self.pipe, negative_prompt="worst quality, low quality, blurry, low resolution, low quality")
        }

        result = self.pipe(
            prompt=input_data["caption"],
            negative_prompt="worst quality, low quality, blurry, low resolution, low quality",
            generator=torch.Generator().manual_seed(input_data['seed']),
            cross_attention_kwargs=cross_attention_kwargs,
            height=input_data['height'],
            width=input_data['width'],
            roictrl_scheduled_sampling_beta=input_data['roictrl_scheduled_sampling_beta']
        ).images[0]


        result.save(save_name)  # save output PIL image

        if self.vis:
            
            img_cv = np.array(result)
            for name in character_box:
                box = character_box[name]

                box[0] = int(box[0] * size[0])
                box[2] = int(box[2] * size[0])
                box[1] = int(box[1] * size[1])
                box[3] = int(box[3] * size[1])
                img_cv = cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
            img_cv = Image.fromarray(img_cv)
            img_cv.save(save_name.replace(".jpg","_vis.jpg"))
