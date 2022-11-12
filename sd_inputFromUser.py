'''Программа принимает ввод
от пользователя на английском языке,
а затем сохраняет изображение .png
с названием, выбранным пользователем
'''

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = input("Введите Ваш вариант описания изображения на английском: ")
how_you_name_file = input("Введите название файла, куда Вы хотите сохранить изображение: ")
with autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5).images[0]  
    
image.save(how_you_name_file + ".png")
