from PIL import Image, ImageDraw, ImageFont
from roictrl.pipelines.pipeline_stable_diffusion_edlora import StableDiffusionEDLoRAPipeline, bind_concept_prompt
from roictrl.models.unet_2d_condition_model import UNet2DConditionModel
import torch
import math
from copy import deepcopy
from safetensors.torch import load_file, save_file
import os
import json
from einops import rearrange
import numpy as np

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

pipe = setup_pipeline(
    pretrained_model_path="/storage/wuweijia/MovieGen/MovieDirector/MovieDirector/tools/Mix-of-Show/experiments/composed_edlora/chilloutmix/NeZha2/combined_model_base",
    roictrl_path="/storage/wuweijia/MovieGen/MovieDirector/ROICtrl/experiments/pretrained_models/ROICtrl_sdv14_30K.safetensors"
)
input_data = {
    "caption": "Close-up of Taiyi and Shenggongbao's faces, Taiyi's expression of determination clashing with Shenggongbao's malice., 4K, high quality, high resolution, best quality, 4K, high quality, high resolution, best quality",
    "roi_boxes": [
        [0.2, 0.4, 0.6, 0.8],
        [0.6, 0.4, 0.9, 0.8]
    ],
    "roi_phrases": [
        "a <Taiyi1> <Taiyi2>",    
        "a <Shenggongbao1> <Shenggongbao2>"
    ],
    "height": 512,
    "width": 1024,
    "seed": 1234,
    "roictrl_scheduled_sampling_beta": 1.0
}

cross_attention_kwargs = {
    'roictrl': encode_roi_input(input_data, pipe, negative_prompt="worst quality, low quality, blurry, low resolution, low quality")
}

# ref_image_paths = ["assets/colab_demo_input/dogB.jpg", "assets/colab_demo_input/catA.jpg", "assets/colab_demo_input/dogA.jpeg"]
# ref_images = [Image.open(path).convert('RGB') for path in ref_image_paths]
# merge_images_horizontally = lambda images: Image.fromarray(np.hstack([np.array(img.resize((512, 512))) for img in ref_images]))
# print("reference customized concepts:")
# merge_images_horizontally(ref_images)

result = pipe(
    prompt=input_data["caption"],
    negative_prompt="worst quality, low quality, blurry, low resolution, low quality",
    generator=torch.Generator().manual_seed(input_data['seed']),
    cross_attention_kwargs=cross_attention_kwargs,
    height=input_data['height'],
    width=input_data['width'],
    roictrl_scheduled_sampling_beta=input_data['roictrl_scheduled_sampling_beta']
).images[0]


save_name = "./test.jpg"
result.save(save_name)  # save output PIL image


# image = np.array(result)
# # print()
# height, width, _ = image.shape
# mid = width // 2  # 水平中点

# 截取左右两部分
# left_image = image[:, :mid]  # 左半部分
# right_image = image[:, mid:]  # 右半部分

# 将截取的图像保存为文件
# left_img = Image.fromarray(left_image)
# right_img = Image.fromarray(right_image)

# 保存结果
# left_img.save("./left_image.png")
# right_img.save("./right_image.png")