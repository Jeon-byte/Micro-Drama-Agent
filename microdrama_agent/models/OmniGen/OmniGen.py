import torch
import numpy as np
import torch
import copy
import os
import random
import datetime
import pdb
from tqdm import tqdm
from PIL import ImageFont


class OminiGen_pipe:
    def __init__(self, model_path="", dtype = "bfloat16",device="cuda"):
        from OmniGen import OmniGenPipeline

        self.pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  



    def predict(self, prompt, refer_images, save_name,seed = 0):

        number = len(refer_images)
        prompt1 = prompt
        for idx,name1 in enumerate(refer_images):
            print(name1)
            real_name = name1.split("/")[-2].replace("_"," ")
            prompt1 += ". The {} is in <img><|image_{}|></img>".format(real_name,idx+1)

        print("------------------")
        print(prompt1)
        upload_images = refer_images
        images = self.pipe(
            prompt=prompt1,
            input_images = upload_images,
            height=720, 
            width=1024,
            guidance_scale=2.5, 
            img_guidance_scale=1.6,
            seed=0
        )
        images[0].save(save_name)  # save output PIL image



