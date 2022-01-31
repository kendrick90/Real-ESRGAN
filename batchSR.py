#@markdown ## **Increase Resolution**

import argparse
import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGAN
import pathlib

parser = argparse.ArgumentParser(description='REAL-ESRGAN run parameters')
# parser.add_argument('--model', type=str, required=True)
parser.add_argument('-i', '--init', type=int, required=True)
parser.add_argument('-e', '--end', type=int, required=True)
parser.add_argument('-f', '--input_folder', type=str, required=True)
parser.add_argument('-s', '--scale', type=int, required=True)
args = parser.parse_args()

device = torch.device('cuda')
model = RealESRGAN(device, scale=args.scale)
model_path = pathlib.Path.cwd() / 'weights' / f"RealESRGAN_x{args.scale}.pth"
model.load_weights(model_path)

path_to_output = pathlib.Path.cwd() / args.input_folder / 'SR'
try:
    path_to_output.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print("SR folder is already there")
else:
    print("SR folder was created")

for i in range(args.init, args.end+1):
    print(f'Upscaling frame {i}')
    filename = f"{i:04}.png"
    path_to_image = pathlib.Path.cwd() / args.input_folder / filename
    image = Image.open(path_to_image).convert('RGB')
    sr_image = model.predict(image)
    path_to_output = pathlib.Path.cwd() / args.input_folder / 'SR' / filename
    sr_image.save(path_to_output)


        
        

        





