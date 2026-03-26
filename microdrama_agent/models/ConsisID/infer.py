import argparse
import os
import random

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video

from .models.consisid_utils import prepare_face_models, process_face_embeddings_infer
from .models.pipeline_consisid import ConsisIDPipeline
from .models.transformer_consisid import ConsisIDTransformer3DModel
from .util.rife_model import load_rife_model, rife_inference_with_latents
from .util.utils import load_sd_upscale, upscale_batch_and_concatenate

def get_random_seed():
    return random.randint(0, 2**32 - 1)

def generate_video(
    prompt: str,
    model_path: str,
    negative_prompt: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    img_file_path: str = None,
    is_upscale: bool = False,
    is_frame_interpolation: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - negative_prompt (str): The description of the negative prompt.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - img_file_path (str): The path of the face image.
    - is_upscale (bool): Whether to apply super-resolution (video upscaling) to the generated video. Default is False.
    - is_frame_interpolation (bool): Whether to perform frame interpolation to increase the frame rate. Default is False.
    """
    # 0. Pre config
    device = "cuda"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if os.path.exists(os.path.join(model_path, "transformer_ema")):
        subfolder = "transformer_ema"
    else:
        subfolder = "transformer"


    # 1. Prepare all the face models
    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models(model_path, device, dtype)


    # 2. Load Pipeline.
    transformer = ConsisIDTransformer3DModel.from_pretrained(model_path, subfolder=subfolder)
    pipe = ConsisIDPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)


    # 3. Move to device.
    transformer.to(device, dtype=dtype)
    pipe.to(device)
    # Save Memory. Turn on if you don't have multiple GPUs or enough GPU memory(such as H100) and it will cost more time in inference, it may also reduce the quality
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()


    # 4. Prepare model input
    id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2,
                                                                            eva_transform_mean, eva_transform_std,
                                                                            face_main_model, device, dtype,
                                                                            img_file_path, is_align_face=True)

    prompt = prompt.strip('"')
    if negative_prompt:
        negative_prompt = negative_prompt.strip('"')


    # 5. Generate Identity-Preserving Video
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    video_pt = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        output_type="pt",
    ).frames

    del pipe
    del transformer
    free_memory()

    if is_upscale:
        print("Upscaling...")
        upscale_model = load_sd_upscale(f"{model_path}/model_real_esran/RealESRGAN_x4.pth", device)
        video_pt = upscale_batch_and_concatenate(upscale_model, video_pt, device)
    if is_frame_interpolation:
        print("Frame Interpolating...")
        frame_interpolation_model = load_rife_model(f"{model_path}/model_rife")
        video_pt = rife_inference_with_latents(frame_interpolation_model, video_pt)

    batch_size = video_pt.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = video_pt[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    # 6. Export the generated frames to a video file. fps must be 8 for original video.
    file_count = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
    video_path = f"{output_path}/{seed}_{file_count:04d}.mp4"
    export_to_video(batch_video_frames[0], video_path, fps=8)


def test():
    prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
    model_path = "ckpts"
    lora_path = None
    lora_rank = 128
    img_file_path = "./2.png"
    negative_prompt = None
    output_path = "./output"
    guidance_scale = 6.0
    num_inference_steps = 50
    num_videos_per_prompt = 1
    dtype = "bfloat16"
    seed = 42

    if not os.path.exists(model_path):
        print("Base Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=model_path)
    else:
        print(f"Base Model already exists in {model_path}, skipping download.")

    generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_path=model_path,
        lora_path=lora_path,
        lora_rank=lora_rank,
        output_path=output_path,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=num_videos_per_prompt,
        dtype=torch.float16 if dtype == "float16" else torch.bfloat16,
        seed=seed,
        img_file_path=img_file_path
    )
