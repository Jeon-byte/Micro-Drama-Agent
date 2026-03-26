from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image
import os
import cv2

class I2Vgen_pipe:
    def __init__(self):
        self.SAMPLE_RATE = 24000
        # download and load all models
        
        self.pipe = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        self.pipe.enable_model_cpu_offload()



    def predict(self, prompt, image_path, video_save_path, size):
        
        # image = load_image(image=image_path)
        image = load_image(image_path).convert("RGB")
        
        image = image.resize(size)
        
        negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
        generator = torch.manual_seed(0)

        frames = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=50,
            negative_prompt=negative_prompt,
            guidance_scale=1.0,
            generator=generator
        ).frames[0]
        export_to_gif(frames, video_save_path.replace("mp4","gif"))


