import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio

# Step 1
render_paths = 'LogiCity_vis/web/vis'  # specify your own path

render_dirs = sorted([d for d in os.listdir(render_paths)])

all_images = {}

print('Loading all images dir...')
for dir in tqdm(render_dirs):
    if '.DS' in dir:
        continue
    num = int(dir.split('_')[-1])
    all_images[num] = []
    for img in sorted(os.listdir(os.path.join(render_paths, dir))):
        if img.endswith('.png'):
            all_images[num].append(os.path.join(render_paths, dir, img))
print('done!')

# Step 2: Convert each folder into a GIF image
output_dir = 'LogiCity_vis/web/gifs'  # specify the output path for GIFs
os.makedirs(output_dir, exist_ok=True)

print('Converting images to GIFs...')
for num, images in tqdm(all_images.items()):
    frames = [imageio.imread(img) for img in images]
    gif_path = os.path.join(output_dir, f'animation_{num}.gif')
    imageio.mimsave(gif_path, frames, duration=0.15)  # adjust duration as needed
print('done!')
