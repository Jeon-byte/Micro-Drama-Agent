from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import os
import cv2

class CogVideoX_pipe:
    def __init__(self):
        self.SAMPLE_RATE = 24000
        # download and load all models
        
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )

        # reduce memory requirements 
        self.pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()
        # self.pipe.vae.enable_tiling()
        # self.pipe.vae.enable_slicing()


    def predict(self, prompt, image_path, video_save_path, size):
        
        # image = load_image(image=image_path)

        video = cv2.VideoCapture(image_path)
        success, image = video.read()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        image = image.resize(size)
        # print(prompt)

        # 修改num_frame可能会损坏mp4
        video = self.pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        export_to_video(video, video_save_path, fps=8)
