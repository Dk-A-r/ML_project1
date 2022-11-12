'''Программа принимает ввод
от пользователя на английском языке,
а затем сохраняет изображение .png
с названием, выбранным пользователем

для того, чтобы программу можно было запустить,
нужно установить необходимые библиотеки,
а также в своем профиле на huggingface
принять условия лицензии
на сайтах: https://huggingface.co/CompVis/stable-diffusion-v1-4
https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

после этого залогиниться: huggingface-cli login
в терминале может быть ошибка, для её устранения
следует запустить терминал от администратора
потом скопировать токен
потом подождать ~5 минут
далее залогиниться (вставка не клавишей, а через правую кнопку
либо контекстное меню терминала
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
