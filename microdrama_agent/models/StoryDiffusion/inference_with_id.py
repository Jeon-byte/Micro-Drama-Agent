import numpy as np
import torch
import copy
import os
import random
import datetime
import pdb
from tqdm import tqdm
from PIL import ImageFont
from .utils.gradio_utils import (
    character_to_dict,
    process_original_prompt,
    get_ref_character,
    cal_attn_mask_xl,
    cal_attn_indice_xl_effcient_memory,
    is_torch2_available,
)

if is_torch2_available():
    from .utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from .utils.gradio_utils import AttnProcessor
from huggingface_hub import hf_hub_download
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch.nn.functional as F
from diffusers.utils.loading_utils import load_image
from .utils.utils import get_comic
from .utils.style_template import styles
from .utils.load_models_utils import get_models_dict, load_models

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"
global models_dict

models_dict = get_models_dict()
print(models_dict)

# Automatically select the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"@@device:{device}")


# check if the file exists locally at a specified path before downloading it.
# if the file doesn't exist, it uses `hf_hub_download` to download the file
# and optionally move it to a specific directory. If the file already
# exists, it simply uses the local path.
local_dir = "data/"
photomaker_local_path = f"{local_dir}photomaker-v1.bin"
if not os.path.exists(photomaker_local_path):
    photomaker_path = hf_hub_download(
        repo_id="TencentARC/PhotoMaker",
        filename="photomaker-v1.bin",
        repo_type="model",
        local_dir=local_dir,
    )
else:
    photomaker_path = photomaker_local_path

MAX_SEED = np.iinfo(np.int32).max


def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#################################################
def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list


#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        device=device,
        dtype=torch.float16,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # un_cond_hidden_states, cond_hidden_states = hidden_states.chunk(2)
        # un_cond_hidden_states = self.__call2__(attn, un_cond_hidden_states,encoder_hidden_states,attention_mask,temb)
        # 生成一个0到1之间的随机数
        global total_count, attn_count, cur_step, indices1024, indices4096
        global sa32, sa64
        global write
        global height, width
        global character_dict, character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals, cur_character
        if attn_count == 0 and cur_step == 0:
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )
        
        # print("------------------------------------------")
        # print(write)
        if write:
            assert len(cur_character) == 1
            if hidden_states.shape[1] == (height // 32) * (width // 32):
                indices = indices1024
            else:
                indices = indices4096
            # print(f"white:{cur_step}")
            total_batch_size, nums_token, channel = hidden_states.shape
            img_nums = total_batch_size // 2
            hidden_states = hidden_states.reshape(-1, img_nums, nums_token, channel)
            # print(img_nums,len(indices),hidden_states.shape,self.total_length)
            if cur_character[0] not in self.id_bank:
                self.id_bank[cur_character[0]] = {}
            self.id_bank[cur_character[0]][cur_step] = [
                hidden_states[:, img_ind, indices[img_ind], :]
                .reshape(2, -1, channel)
                .clone()
                for img_ind in range(img_nums)
            ]
            hidden_states = hidden_states.reshape(-1, nums_token, channel)
            # self.id_bank[cur_step] = [hidden_states[:self.id_length].clone(), hidden_states[self.id_length:].clone()]
        else:
            # encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),self.id_bank[cur_step][1].to(self.device)))
            # TODO: ADD Multipersion Control
            
            encoder_arr = []
            for character in cur_character:
                encoder_arr = encoder_arr + [
                    tensor.to(self.device)
                    for tensor in self.id_bank[character][cur_step]
                ]
        # 判断随机数是否大于0.5
        if cur_step < 1:
            hidden_states = self.__call2__(
                attn, hidden_states, None, attention_mask, temb
            )
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            # print(f"hidden state shape {hidden_states.shape[1]}")
            if random_number > rand_num:
                if hidden_states.shape[1] == (height // 32) * (width // 32):
                    indices = indices1024
                else:
                    indices = indices4096
                # print("before attention",hidden_states.shape,attention_mask.shape,encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
                if write:
                    total_batch_size, nums_token, channel = hidden_states.shape
                    img_nums = total_batch_size // 2
                    hidden_states = hidden_states.reshape(
                        -1, img_nums, nums_token, channel
                    )
                    encoder_arr = [
                        hidden_states[:, img_ind, indices[img_ind], :].reshape(
                            2, -1, channel
                        )
                        for img_ind in range(img_nums)
                    ]
                    for img_ind in range(img_nums):
                        # print(img_nums)
                        # assert img_nums != 1
                        img_ind_list = [i for i in range(img_nums)]
                        # print(img_ind_list,img_ind)
                        img_ind_list.remove(img_ind)
                        # print(img_ind,invert_character_index_dict[img_ind])
                        # print(character_index_dict[invert_character_index_dict[img_ind]])
                        # print(img_ind_list)
                        # print(img_ind,img_ind_list)
                        encoder_hidden_states_tmp = torch.cat(
                            [encoder_arr[img_ind] for img_ind in img_ind_list]
                            + [hidden_states[:, img_ind, :, :]],
                            dim=1,
                        )

                        hidden_states[:, img_ind, :, :] = self.__call2__(
                            attn,
                            hidden_states[:, img_ind, :, :],
                            encoder_hidden_states_tmp,
                            None,
                            temb,
                        )
                else:
                    _, nums_token, channel = hidden_states.shape
                    # img_nums = total_batch_size // 2
                    # encoder_hidden_states = encoder_hidden_states.reshape(-1,img_nums,nums_token,channel)
                    hidden_states = hidden_states.reshape(2, -1, nums_token, channel)
                    # print(len(indices))
                    # encoder_arr = [encoder_hidden_states[:,img_ind,indices[img_ind],:].reshape(2,-1,channel) for img_ind in range(img_nums)]
                    encoder_hidden_states_tmp = torch.cat(
                        encoder_arr + [hidden_states[:, 0, :, :]], dim=1
                    )
                    # print(len(encoder_arr),encoder_hidden_states_tmp.shape)
                    hidden_states[:, 0, :, :] = self.__call2__(
                        attn,
                        hidden_states[:, 0, :, :],
                        encoder_hidden_states_tmp,
                        None,
                        temb,
                    )
                hidden_states = hidden_states.reshape(-1, nums_token, channel)
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )

        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        # else:
        #     encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attention_processor(unet, id_length, is_ipadapter=False):
    global attn_procs
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet.device, dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))


#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = """
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
"""


def save_single_character_weights(unet, character, description, filepath):
    """
    保存 attention_processor 类中的 id_bank GPU Tensor 列表到指定文件中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重要保存到的文件路径。
    """
    weights_to_save = {}
    weights_to_save["description"] = description
    weights_to_save["character"] = character
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 将每个 Tensor 转到 CPU 并转为列表，以确保它可以被序列化
            weights_to_save[attn_name] = {}
            for step_key in attn_processor.id_bank[character].keys():
                weights_to_save[attn_name][step_key] = [
                    tensor.cpu()
                    for tensor in attn_processor.id_bank[character][step_key]
                ]
    # 使用torch.save保存权重
    torch.save(weights_to_save, filepath)


def load_single_character_weights(unet, filepath):
    """
    从指定文件中加载权重到 attention_processor 类的 id_bank 中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重文件的路径。
    """
    # 使用torch.load来读取权重
    weights_to_load = torch.load(filepath, map_location=torch.device("cpu"))
    character = weights_to_load["character"]
    description = weights_to_load["description"]
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 转移权重到GPU（如果GPU可用的话）并赋值给id_bank
            attn_processor.id_bank[character] = {}
            for step_key in weights_to_load[attn_name].keys():
                attn_processor.id_bank[character][step_key] = [
                    tensor.to(unet.device)
                    for tensor in weights_to_load[attn_name][step_key]
                ]


# def save_results(unet, img_list):

#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     folder_name = f"results/{timestamp}"
#     weight_folder_name = f"{folder_name}/weights"
#     # 创建文件夹
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#         os.makedirs(weight_folder_name)

#     for idx, img in enumerate(img_list):
#         file_path = os.path.join(folder_name, f"image_{idx}.png")  # 图片文件名
#         img.save(file_path)
#     global character_dict
    # for char in character_dict:
    #     description = character_dict[char]
    #     save_single_character_weights(unet,char,description,os.path.join(weight_folder_name, f'{char}.pt'))


#################################################
title = r"""
<h1 align="center">StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'><b>StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</b></a>.<br>
❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
1️⃣ Enter a Textual Description for Character, if you add the Ref-Image, making sure to <b>follow the class word</b> you want to customize with the <b>trigger word</b>: `img`, such as: `man img` or `woman img` or `girl img`.<br>
2️⃣ Enter the prompt array, each line corrsponds to one generated image.<br>
3️⃣ Choose your preferred style template.<br>
4️⃣ Click the <b>Submit</b> button to start customizing.
"""

article = r"""

If StoryDiffusion is helpful, please help to ⭐ the <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/HVision-NKU/StoryDiffusion?style=social)](https://github.com/HVision-NKU/StoryDiffusion)
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:

```bibtex
@article{Zhou2024storydiffusion,
  title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
  author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
  year={2024}
}
```
📋 **License**
<br>
Apache-2.0 LICENSE. 

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>ypzhousdu@gmail.com</b>.
"""
version = r"""
<h3 align="center">StoryDiffusion Version 0.02 (test version)</h3>

<h5 >1. Support image ref image. (Cartoon Ref image is not support now)</h5>
<h5 >2. Support Typesetting Style and Captioning.(By default, the prompt is used as the caption for each image. If you need to change the caption, add a # at the end of each line. Only the part after the # will be added as a caption to the image.)</h5>
<h5 >3. [NC]symbol (The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want do that, prepend the "[NC]" at the beginning of the line. For example, to generate a scene of falling leaves without any character, write: "[NC] The leaves are falling.")</h5>
<h5 align="center">Tips: </h4>
"""
#################################################
global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
global write
global sa32, sa64
global height, width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
global attn_procs, unet
attn_procs = {}
###
write = False
###
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
###
global pipe
global sd_model_path
pipe = None
sd_model_path = models_dict["Unstable"]["path"]  # "SG161222/RealVisXL_V4.0"
single_files = models_dict["Unstable"]["single_files"]
### LOAD Stable Diffusion Pipeline
if single_files:
    pipe = StableDiffusionXLPipeline.from_single_file(
        sd_model_path, torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sd_model_path, torch_dtype=torch.float16, use_safetensors=False
    )
pipe = pipe.to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
pipe.enable_vae_slicing()
if device != "mps":
    pipe.enable_model_cpu_offload()
unet = pipe.unet
cur_model_type = "Unstable" + "-" + "original"
### Insert PairedAttention
for name in unet.attn_processors.keys():
    cross_attention_dim = (
        None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    )
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks")):
        attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
        total_count += 1
    else:
        attn_procs[name] = AttnProcessor()
print("successsfully load paired self-attention")
print(f"number of the processor : {total_count}")
unet.set_attn_processor(copy.deepcopy(attn_procs))
global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(
    total_length,
    id_length,
    sa32,
    sa64,
    height,
    width,
    device=device,
    dtype=torch.float16,
)

######### Gradio Fuction #############
def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative

def load_character_files_on_running(unet, character_files: str):
    if character_files == "":
        return False
    character_files_arr = character_files.splitlines()
    for character_file in character_files_arr:
        load_single_character_weights(unet, character_file)
    return True

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

######### Image Generation ##############
def process_generation(
    _sd_type,
    _model_type,
    _upload_images,
    _num_steps,
    style_name,
    _Ip_Adapter_Strength,
    _style_strength_ratio,
    guidance_scale,
    seed_,
    sa32_,
    sa64_,
    id_length_,
    general_prompt,
    negative_prompt,
    prompt_array,
    G_height,
    G_width,
    _comic_type,
    font_choice,
    _char_files
):  # Corrected font_choice usage
    if len(general_prompt.splitlines()) >= 3:
        print("Support for more than three characters is temporarily unavailable due to VRAM limitations, but this issue will be resolved soon." )
    _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    if _model_type == "Photomaker" and "img" not in general_prompt:
         print('Please add the triger word " img "  behind the class word you want to customize, such as: man img or woman img')
    if _upload_images is None and _model_type != "original":
         print(f"Cannot find any input face image!")
    global sa32, sa64, id_length, total_length, attn_procs, unet, cur_model_type
    global write
    global cur_step, attn_count
    global height, width
    height = G_height
    width = G_width
    global pipe
    global sd_model_path, models_dict
    sd_model_path = models_dict[_sd_type]
    
    use_safe_tensor = True
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            for values in attn_processor.id_bank.values():
                del values
            attn_processor.id_bank = {}
            attn_processor.id_length = id_length
            attn_processor.total_length = id_length + 1
    torch.cuda.empty_cache()

    # print(cur_model_type)
    # assert False
    if cur_model_type != _sd_type + "-" + _model_type:
        # apply the style template
        ##### load pipe
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        model_info = models_dict[_sd_type]
        model_info["model_type"] = _model_type
        pipe = load_models(model_info, device=device, photomaker_path=photomaker_path)
        set_attention_processor(pipe.unet, id_length_, is_ipadapter=False)
        ##### ########################
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        cur_model_type = _sd_type + "-" + _model_type
        pipe.enable_vae_slicing()
        if device != "mps":
            pipe.enable_model_cpu_offload()
    else:
        unet = pipe.unet
        # unet.set_attn_processor(copy.deepcopy(attn_procs))
    # pdb.set_trace()
    
    load_chars = load_character_files_on_running(unet, character_files=_char_files)
    prompts = prompt_array.splitlines()
    global character_dict, character_index_dict, invert_character_index_dict, ref_indexs_dict, ref_totals
    character_dict, character_list = character_to_dict(general_prompt)
    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    generator = torch.Generator(device=device).manual_seed(seed_)
    sa32, sa64 = sa32_, sa64_
    id_length = id_length_
    clipped_prompts = prompts[:]
    nc_indexs = []
    for ind, prompt in enumerate(clipped_prompts):
        if "[NC]" in prompt:
            nc_indexs.append(ind)
            if ind < id_length:
                print(f"The first {id_length} row is id prompts, cannot use [NC]!")
    prompts = [
        prompt if "[NC]" not in prompt else prompt.replace("[NC]", "")
        for prompt in clipped_prompts
    ]

    prompts = [
        prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
    ]
    # print(prompts)
    # id_prompts = prompts[:id_length]
    (
        character_index_dict,
        invert_character_index_dict,
        replace_prompts,
        ref_indexs_dict,
        ref_totals,
    ) = process_original_prompt(character_dict, prompts.copy(), id_length)
    if _model_type != "original":
        input_id_images_dict = {}
        if len(_upload_images) != len(character_dict.keys()):
            print(f"You upload images({len(_upload_images)}) is not equal to the number of characters({len(character_dict.keys())})!"
)
        for ind, img in enumerate(_upload_images):
            input_id_images_dict[character_list[ind]] = [load_image(img)]
    # print("------****-----")
    # print(character_dict)
    # print(character_index_dict)
    # print(invert_character_index_dict)
    # real_prompts = prompts[id_length:]
    if device == "cuda":
        torch.cuda.empty_cache()
    write = True
    cur_step = 0

    attn_count = 0
    # id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    # print(id_prompts)
    setup_seed(seed_)
    total_results = []
    id_images = []
    results_dict = {}
    global cur_character
    if not load_chars:
        for character_key in character_dict.keys():
            cur_character = [character_key]

            ref_indexs = ref_indexs_dict[character_key]
            # print(character_key, ref_indexs)
            
            current_prompts = [replace_prompts[ref_ind] for ref_ind in ref_indexs]
            # print(current_prompts)
            setup_seed(seed_)
            generator = torch.Generator(device=device).manual_seed(seed_)
            cur_step = 0
            cur_positive_prompts, negative_prompt = apply_style(
                style_name, current_prompts, negative_prompt
            )
            if _model_type == "original":
                id_images = pipe(
                    cur_positive_prompts,
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            elif _model_type == "Photomaker":
                id_images = pipe(
                    cur_positive_prompts,
                    input_id_images=input_id_images_dict[character_key],
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    start_merge_step=start_merge_step,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            else:
                raise NotImplementedError(
                    "You should choice between original and Photomaker!",
                    f"But you choice {_model_type}",
                )

            # total_results = id_images + total_results
            # yield total_results
            # print(id_images)
            for ind, img in enumerate(id_images):
                # print(ref_indexs[ind])
                results_dict[ref_indexs[ind]] = img
            # assert False
            # real_images = []
    # pdb.set_trace()
    
    write = False
    if not load_chars:
        real_prompts_inds = [
            ind for ind in range(len(prompts)) if ind not in ref_totals
        ]
    else:
        real_prompts_inds = [ind for ind in range(len(prompts))]
    # print(real_prompts_inds)
    
    # print(load_chars)
    # print(_model_type)
    # assert False
    for real_prompts_ind in tqdm(real_prompts_inds):
        real_prompt = replace_prompts[real_prompts_ind]
        cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        # print(cur_character, real_prompt)
        setup_seed(seed_)
        if len(cur_character) > 1 and _model_type == "Photomaker":
            print("Temporarily Not Support Multiple character in Ref Image Mode!")
        generator = torch.Generator(device=device).manual_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        print('forward')
        if _model_type == "original":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        elif _model_type == "Photomaker":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                input_id_images=(
                    input_id_images_dict[cur_character[0]]
                    if real_prompts_ind not in nc_indexs
                    else input_id_images_dict[character_list[0]]
                ),
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                start_merge_step=start_merge_step,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
                nc_flag=True if real_prompts_ind in nc_indexs else False,
            ).images[0]
        else:
            raise NotImplementedError(
                "You should choice between original and Photomaker!",
                f"But you choice {_model_type}",
            )

    # print([i for i in results_dict.keys()])
    # assert False
    total_results = [results_dict[ind] for ind in range(len(prompts))]
    if _comic_type != "No typesetting (default)":
        captions = prompt_array.splitlines()
        captions = [caption.replace("[NC]", "") for caption in captions]
        captions = [
            caption.split("#")[-1] if "#" in caption else caption
            for caption in captions
        ]
        font_path = os.path.join("models/StoryDiffusion/fonts", font_choice)
        font = ImageFont.truetype(font_path, int(45))
        total_results = (
            get_comic(total_results, _comic_type, captions=captions, font=font)
            + total_results
        )
    return total_results


def array2string(arr):
    stringtmp = ""
    for i, part in enumerate(arr):
        if i != len(arr) - 1:
            stringtmp += part + "\n"
        else:
            stringtmp += part

    return stringtmp


# G_height, G_width = 1024, 1024
# font_choice = 'Inkfree.ttf'
# char_path = ''

# if __name__ == '__main__':
#     sd_type, model_type = "Unstable", "Using Ref Images"
#     _upload_images = get_image_path_list("/users/zeyuzhu/movie_gen/StoryDiffusion/examples/JM")
#     sa32_, sa64_ = 0.5, 0.5 # network
#     num_steps, guidance_scale = 35, 5.0 # diffusion
#     Ip_Adapter_Strength, style_strength_ratio = 0.5, 20 # module weight
    
#     seed = 0
#     id_length = 2
#     ## prompt for the characters
#     general_prompt = "[Juno MacGuff]A woman img"
#     negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs, sexical content"
#     ## prompt for the stories
    
#     # prompt_array =  array2string([
#     #                     "[Bob] at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
#     #                     "[Bob] on the road, near the forest",
#     #                     "[Alice] is make a call at home # [Bob] invited [Alice] to join him on an adventure.",
#     #                     "[NC] The car on the road, near the forest #They drives to the forest in search of treasure.",
#     #                     "[NC]A tiger appeared in the forest, at night ",
#     #                     "[Bob] very frightened, open mouth, in the forest, at night",
#     #                     "[Alice] very frightened, open mouth, in the forest, at night",
#     #                     "[Bob]  running very fast, in the forest, at night",
#     #                     "[NC] A house in the forest, at night #Suddenly, They discovers the treasure house!",
#     #                     "[Bob]  in the house filled with  treasure, laughing, at night #They are overjoyed inside the house.",
#     #                 ])
#     prompt_array =  array2string([
#                     "[NC] The scene shows a low-rise bungalow perched on a gentle slope, overlooking an uneven lawn under a cloudy sky. The word 'AUTUMN' appears in large letters, indicating the season.",
#                     "[Juno MacGuff] Teenage girl with short brown hair tied back in a small ponytail, swigs from a plastic bottle. she stands on a lawn, swigging from a small plastic bottle while contemplating an armchair placed in front of her. The scene captures a moment of quiet reflection in a suburban setting during autumn.",
#                     "[Juno MacGuff] In a flashback, Juno MacGuff steps out of her knickers and walks slowly across the room towards Paulie Bleeker, who is sitting bare-chested in an armchair. The scene suggests an intimate moment between the two characters."
#                     ])
#     style_name, style = 'Photographic', 'Photographic' # image style
#     comic_type = ''
    
#     args_list = [sd_type, model_type, _upload_images, num_steps, style_name,
#                 Ip_Adapter_Strength, style_strength_ratio, guidance_scale,
#                 seed, sa32_, sa64_, id_length,
#                 general_prompt, negative_prompt, prompt_array,
#                 G_height, G_width,
#                 comic_type,
#                 font_choice, char_path]
#     process_generation(*args_list)
# import os
# import pdb
# import json
# import numpy as np
# from os import path

# def save_results(save_folder, img_list, img_name):
#     folder_name = save_folder
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     for idx, img in enumerate(img_list):
#         file_path = os.path.join(folder_name, f"{img_name[idx]}.png")
#         img.save(file_path)
    
# def construct_args_run(save_folder, img_list, upload_images, general_prompt, prompt_array):
#     sd_type, model_type = "Unstable", "Using Ref Images"
#     _upload_images = get_image_path_list("/users/zeyuzhu/movie_gen/StoryDiffusion/examples/JM")
#     sa32_, sa64_ = 0.5, 0.5 # network
#     num_steps, guidance_scale = 35, 5.0 # diffusion
#     Ip_Adapter_Strength, style_strength_ratio = 0.5, 20 # module weight
    
#     seed = 0
#     id_length = 2
#     ## prompt for the characters
#     prompt_array = array2string(prompt_array)
#     negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs, sexical content"
#     style_name, style = 'Photographic', 'Photographic' # image style
#     comic_type = ''
#     G_height, G_width = 1024, 1024
#     font_choice = 'Inkfree.ttf'
#     char_path = ''
#     args_list = [sd_type, model_type, upload_images, num_steps, style_name,
#                 Ip_Adapter_Strength, style_strength_ratio, guidance_scale,
#                 seed, sa32_, sa64_, id_length,
#                 general_prompt, negative_prompt, prompt_array,
#                 G_height, G_width,
#                 comic_type,
#                 font_choice, char_path]
#     result = process_generation(*args_list)
    
#     pdb.set_trace()
#     save_results(save_folder, result, img_list)
    
# def inference(data):
#     story_root, img_root, save_root = data[0], data[1], data[2]
#     ## load the stories
#     story_dic = {}
#     for file_name in os.listdir(story_root):
#         with open(path.join(story_root, file_name), 'r') as f:
#             story_dic[file_name] = json.load(f)
#     ## load the image_path
#     img_dic = {}
#     for img_name in os.listdir(img_root):
#         char_name = img_name.split('-')[-1].replace('_', '').replace(' ', '') # get char name
#         img_dic[char_name] = path.join(img_root, img_name, 'best.jpg')
#     ## align character with stories
#     char_story_dic = {}
#     char_story_dic['NC'] = []
#     for file_name in story_dic.keys():
#         story = story_dic[file_name]
#         char_list = story['Characters']
#         if len(char_list) == 0: # no character, belong to NC
#             char_story_dic['NC'].append([file_name, story])
#         else:
#             save_flag = False
#             for char in char_list:
#                 char = char.replace('_', '').replace(' ', '')
#                 if char in img_dic.keys():
#                     if char not in char_story_dic.keys(): 
#                         char_story_dic[char] = []
#                     char_story_dic[char].append([file_name, story])
#                     save_flag = True
#                     break # only control one character
#             if save_flag is False: # no character img, belong to NC
#                 char_story_dic['NC'].append([file_name, story])
#     ## inference for each character
#     for char_name in char_story_dic.keys():
#         if char_name == 'NC':
#             continue
#         char_name = 'JunoMacGuff'
#         print(img_dic.keys())
#         story_list = char_story_dic[char_name]
#         img_name = []
#         prompt_array = []
#         upload_images = [img_dic[char_name]]
#         general_prompt = '[{}], a {} img'.format(char_name, 'woman')
#         for story in story_list:
#             story_line = '[{}] '.format(char_name)+story[1]['Plot']
#             prompt_array.append(story_line)
#             img_name.append(story[0])
#             if len(prompt_array) > 1:
#                 break
#         # prompt_array =  [
#         #         "[NC] The scene shows a low-rise bungalow perched on a gentle slope, overlooking an uneven lawn under a cloudy sky. The word 'AUTUMN' appears in large letters, indicating the season.",
#         #         "[JunoMacGuff] Teenage girl with short brown hair tied back in a small ponytail, swigs from a plastic bottle. she stands on a lawn, swigging from a small plastic bottle while contemplating an armchair placed in front of her. The scene captures a moment of quiet reflection in a suburban setting during autumn.",
#         #         "[JunoMacGuff] In a flashback, Juno MacGuff steps out of her knickers and walks slowly across the room towards Paulie Bleeker, who is sitting bare-chested in an armchair. The scene suggests an intimate moment between the two characters."
#         #     ]
#         pdb.set_trace()
#         result = construct_args_run(save_root, img_name, upload_images, general_prompt, prompt_array)
#         save_results(save_root, result, img_name)

# def limit_words(text, word_limit=30):
#     words = text.split() 
#     if len(words) > word_limit:
#         return ' '.join(words[:word_limit])
#     return text
 
# if __name__ == '__main__':
#     save_root_path = ' /users/zeyuzhu/movie_gen/StoryDiffusion/test_result'
#     story_root_path = '/storage/wuweijia/MovieGen/lsmdc/Annotation_Shot_Desc/Test'
#     identity_root_path = '/storage/wuweijia/MovieGen/lsmdc/GT/Character_Bank/Char_Bank/Test'
#     test_data_list = ['1004_Juno', '1017_Bad_Santa', '1027_Les_Miserables', '1040_The_Ugly_Truth', 
#                       '1041_This_is_40', '1054_Harry_Potter_and_the_prisoner_of_azkaban']
#     inference([path.join(story_root_path, test_data_list[0]),
#               path.join(identity_root_path, test_data_list[0]),
#               path.join(save_root_path, test_data_list[0])])
    