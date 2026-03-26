import torch
# from diffusers import ConsisIDPipeline
# from diffusers.pipelines.consisid.consisid_utils import prepare_face_models, process_face_embeddings_infer

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

from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from .infer import test

class ConsisID:
    def __init__(self, model_path="/storage/wuweijia/MovieGen/MovieDirector/MovieDirector/movie_agent/ckpts", dtype = "bfloat16"):
        self.device = "cuda"
        self.model_path = model_path
        self.is_upscale = False
        self.guidance_scale = 6.0
        self.num_inference_steps = 50
        self.num_videos_per_prompt = 1
        self.is_frame_interpolation = False
        self.negative_prompt = None
        self.dtype=torch.float16 if dtype == "float16" else torch.bfloat16
        lora_path = None
        lora_rank = 128

        if os.path.exists(os.path.join(model_path, "transformer_ema")):
            subfolder = "transformer_ema"
        else:
            subfolder = "transformer"

        # 1. Prepare all the face models
        self.face_helper_1, self.face_helper_2, self.face_clip_model, self.face_main_model, self.eva_transform_mean, self.eva_transform_std = \
            prepare_face_models(model_path, self.device, self.dtype)

        # 2. Load Pipeline.
        # print(model_path)
        self.transformer = ConsisIDTransformer3DModel.from_pretrained(model_path, subfolder=subfolder)
        self.pipe = ConsisIDPipeline.from_pretrained(model_path, torch_dtype=self.dtype)

        # If you're using with lora, add this code
        if lora_path:
            self.pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
            self.pipe.fuse_lora(lora_scale=1 / lora_rank)

        # 3. Move to device.
        self.transformer.to(self.device, dtype=self.dtype)
        self.pipe.to(self.device)
        
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_sequential_cpu_offload()
    
        pass
        # snapshot_download(repo_id=model_name, local_dir=model_name)
        # self.face_helper_1, self.face_helper_2, self.face_clip_model, self.face_main_model, self.eva_transform_mean, self.eva_transform_std = (
        #     prepare_face_models(model_name, device="cuda", dtype=torch.bfloat16)
        # )
        # self.pipe = ConsisIDPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # self.pipe.to("cuda")


    def predict(self, prompt, refer_images, save_name):
        refer_image = refer_images[0]

        # print(refer_image)
        # 4. Prepare model input
        id_cond, id_vit_hidden, image, face_kps = \
            process_face_embeddings_infer(self.face_helper_1, self.face_clip_model, self.face_helper_2,
                                            self.eva_transform_mean, self.eva_transform_std,
                                            self.face_main_model, self.device, self.dtype,
                                            refer_image, is_align_face=True)
        
        prompt = prompt.strip('"')
        if self.negative_prompt:
            self.negative_prompt = self.negative_prompt.strip('"')

        # 5. Generate Identity-Preserving Video
        seed = 42
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None
        video_pt = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            num_videos_per_prompt=self.num_videos_per_prompt,
            num_inference_steps=self.num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=False,
            guidance_scale=self.guidance_scale,
            generator=generator,
            id_vit_hidden=id_vit_hidden,
            id_cond=id_cond,
            kps_cond=face_kps,
            output_type="pt",
        ).frames

        # del self.pipe
        # del transformer
        free_memory()

        if self.is_upscale:
            print("Upscaling...")
            upscale_model = load_sd_upscale(f"{self.model_path}/model_real_esran/RealESRGAN_x4.pth", self.device)
            video_pt = upscale_batch_and_concatenate(upscale_model, video_pt, self.device)
        if self.is_frame_interpolation:
            print("Frame Interpolating...")
            frame_interpolation_model = load_rife_model(f"{self.model_path}/model_rife")
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
        export_to_video(batch_video_frames[0], save_name, fps=8)

        # test()
        # id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
        #     self.face_helper_1,
        #     self.face_clip_model,
        #     self.face_helper_2,
        #     self.eva_transform_mean,
        #     self.eva_transform_std,
        #     self.face_main_model,
        #     "cuda",
        #     torch.bfloat16,
        #     refer_image,
        #     is_align_face=True,
        # )

        # video = self.pipe(
        #     image=image,
        #     prompt=prompt,
        #     num_inference_steps=50,
        #     guidance_scale=6.0,
        #     use_dynamic_cfg=False,
        #     id_vit_hidden=id_vit_hidden,
        #     id_cond=id_cond,
        #     kps_cond=face_kps,
        #     generator=torch.Generator("cuda").manual_seed(42),
        # )
        
        # export_to_video(video.frames[0], save_name, fps=8)