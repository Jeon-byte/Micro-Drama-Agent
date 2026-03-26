from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import os
import cv2

class SVD_pipe:
    def __init__(self):
        self.SAMPLE_RATE = 24000
        # download and load all models
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
        )
        self.pipe.enable_model_cpu_offload()

    def predict(self, prompt, image_path, video_save_path, size = (569, 320)):
        
        # image = load_image(image=image_path)

        video = cv2.VideoCapture(image_path)
        success, image = video.read()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        image = image.resize(size)

        generator = torch.manual_seed(42)
        frames = self.pipe(image, decode_chunk_size=8, generator=generator, num_frames=25).frames[0]
        frames = [i.resize(size) for i in frames]
        # print(frames[0].size)
        export_to_video(frames, video_save_path, fps=7)

