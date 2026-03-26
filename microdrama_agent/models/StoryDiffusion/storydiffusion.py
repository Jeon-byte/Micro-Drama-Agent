import torch
# from diffusers import ConsisIDPipeline
# from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer

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
from .inference_with_id import process_generation, array2string

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

import torch

attn_count = 0 
cur_step = 0
id_length = 4
total_length = 5

sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
write = False

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


def load_character_files_on_running(unet, character_files: str):
    if character_files == "":
        return False
    character_files_arr = character_files.splitlines()
    for character_file in character_files_arr:
        load_single_character_weights(unet, character_file)
    return True


def save_results(save_folder, result, img_name):
    folder_name = save_folder
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for idx, img in enumerate(result):
        if idx == 0: continue
        file_path = os.path.join(folder_name, f"{img_name}.png")
        img.save(file_path)

def set_attention_processor(unet, id_length,device, is_ipadapter=False):
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
        device="cuda",
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
            # print(cur_character[0])
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
            # print(self.id_bank.keys())
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


def setup_seed(seed,device):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def apply_style_positive(style_name: str, positive: str, DEFAULT_STYLE_NAME = "Japanese Anime"):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

class StoryDiffusion:
    def __init__(self, model_path="", dtype = "bfloat16",device="cuda"):
        self.style_name, self.style = 'Photographic', 'Photographic' # image style
        self.id_length = 1
        self.num_steps, self.guidance_scale = 35, 5.0 # diffusion
        self.Ip_Adapter_Strength, self.style_strength_ratio = 0.5, 20 # module weight
        self.negative_prompt = "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs"
        self.comic_type = ''
        self.height, self.width = 864, 1536
        self.font_choice = 'Inkfree.ttf'

        self.start_merge_step = int(float(self.style_strength_ratio) / 100 * self.num_steps)
        if self.start_merge_step > 30:
            self.start_merge_step = 30

        self.sd_type, model_type = "Unstable", "Using Ref Images"
        models_dict = get_models_dict()
        if device == "cuda":
            torch.cuda.empty_cache()
        global write
        write = True
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

        model_info = models_dict[self.sd_type]
        self.model_type = "Photomaker" if model_type == "Using Ref Images" else "original"
        model_info["model_type"] = self.model_type

        # load unet
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


        # pipe
        self.pipe = load_models(model_info, device=device, photomaker_path=photomaker_path)
        set_attention_processor(self.pipe.unet, self.id_length, device, is_ipadapter=False)

        self.pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        cur_model_type = self.sd_type + "-" + self.model_type
        self.pipe.enable_vae_slicing()
        if device != "mps":
            self.pipe.enable_model_cpu_offload()
        char_path = ''
        self.load_chars = load_character_files_on_running(unet, character_files=char_path)

        self.device = device
        pass


        
    def predict(self, prompt, refer_images, character_box, save_name, size, seed = 0):
        from .run_weijia import construct_args_run

        # save_root = "./test"
        try:
            if not os.path.exists(refer_images[0]):
                upload_images = ["./blank.png"]
                character_box = ["None"]
            else:
                upload_images = [refer_images[0]]
        except:
            upload_images = ["./blank.png"]
            character_box = ["None"]
        
        # name = ""
        for name1 in character_box:
            name = name1
            break

        # if name != "":
        general_prompt = "[{}], a human img".format(name)
        if name in prompt:
            prompt1 = prompt.replace(name,"[{}]".format(name), 1)
        else:
            prompt1 = "[{}]".format(name) + prompt
        # else:
        #     general_prompt = ""
        #     prompt1 = ""

        prompt_array = [prompt1]


        # print("-----------------------------")
        # print(prompt_array, prompt, refer_images, save_name)
        # assert False

        print("------------------------")
        print(prompt_array)
        print(upload_images)
        save_root, img_name = os.path.split(save_name) 
        construct_args_run(save_root, img_name, upload_images, general_prompt, prompt_array)

        # save_root = "./test"
        # img_name= "test"
        # upload_images = ['/storage/wuweijia/MovieGen/lsmdc/GT/Character_Bank/Char_Bank/Test/1004_Juno/Jason_Bateman-Mark_Loring/best.jpg']
        # general_prompt = "[MarkLoring], a human img"
        # prompt_array = ['[MarkLoring] Mark opens the front door and catches sight of Juno driving off, indicating a moment of departure or farewell.']

        # prompt_array = array2string(prompt_array)
        # prompts = prompt_array.splitlines()
        # character_dict, character_list = character_to_dict(general_prompt)
        
        # id_length = self.id_length
        # clipped_prompts = prompts[:]
        # nc_indexs = []
        # for ind, prompt in enumerate(clipped_prompts):
        #     if "[NC]" in prompt:
        #         nc_indexs.append(ind)
        #         if ind < id_length:
        #             print(f"The first {id_length} row is id prompts, cannot use [NC]!")
        # prompts = [
        #     prompt if "[NC]" not in prompt else prompt.replace("[NC]", "")
        #     for prompt in clipped_prompts
        # ]

        # (
        #     character_index_dict,
        #     invert_character_index_dict,
        #     replace_prompts,
        #     ref_indexs_dict,
        #     ref_totals,
        # ) = process_original_prompt(character_dict, prompts.copy(), self.id_length)

        # prompts = [
        #     prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
        # ]

        # input_id_images_dict = {}
        # for ind, img in enumerate(upload_images):
        #     input_id_images_dict[character_list[ind]] = [load_image(img)]

        # if not self.load_chars:
        #     # if ind not in ref_totals
        #     real_prompts_inds = [
        #         ind for ind in range(len(prompts))
        #     ]
        #     # real_prompts_inds = prompts


        # else:
        #     real_prompts_inds = [ind for ind in range(len(prompts))]

        # global cur_character
        # for character_key in character_dict.keys():
        #     cur_character = [character_key]
            
        # results_dict = {}
        # for real_prompts_ind in tqdm(real_prompts_inds):
        #     real_prompt = replace_prompts[real_prompts_ind]
        #     cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        #     setup_seed(seed,self.device)
        #     if len(cur_character) > 1 and self.model_type == "Photomaker":
        #         print("Temporarily Not Support Multiple character in Ref Image Mode!")
        #     generator = torch.Generator(device=self.device).manual_seed(seed)
        #     cur_step = 0
        #     real_prompt = apply_style_positive(self.style_name, real_prompt)
            
        #     results_dict[real_prompts_ind] = self.pipe(
        #             real_prompt,
        #             input_id_images=(
        #                 input_id_images_dict[cur_character[0]]
        #                 if real_prompts_ind not in nc_indexs
        #                 else input_id_images_dict[character_list[0]]
        #             ),
        #             num_inference_steps=self.num_steps,
        #             guidance_scale=self.guidance_scale,
        #             start_merge_step=self.start_merge_step,
        #             height=self.height,
        #             width=self.width,
        #             negative_prompt=self.negative_prompt,
        #             generator=generator,
        #             nc_flag=True if real_prompts_ind in nc_indexs else False,
        #         ).images[0]
        
        
        # total_results = [results_dict[ind] for ind in range(len(prompts))]

        # if self.comic_type != "No typesetting (default)":
        #     captions = prompt_array.splitlines()
        #     captions = [caption.replace("[NC]", "") for caption in captions]
        #     captions = [
        #         caption.split("#")[-1] if "#" in caption else caption
        #         for caption in captions
        #     ]
        #     font_path = os.path.join("models/StoryDiffusion/fonts", self.font_choice)
        #     font = ImageFont.truetype(font_path, int(45))
        #     total_results = (
        #         get_comic(total_results, self.comic_type, captions=captions, font=font)
        #         + total_results
        #     )
        # save_results(save_root, total_results, img_name)

        # construct_args_run(save_root, img_name, upload_images, general_prompt, prompt_array)



