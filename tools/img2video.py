import cv2
import os
from glob import glob

image_folder = 'vis_city'

image_files = glob(os.path.join(image_folder, "*.png"))
image_files.sort()
n_images = len(image_files)

from PIL import Image

# set the file names and output file name
output_file = "vis_easy_2k.gif"

# open the first image
with Image.open(image_files[0]) as im:
    # save the first image as the animated GIF
    im.save(output_file, save_all=True, \
        append_images=[Image.open(os.path.join(image_folder, "step_{}.png".format(k))) for k in range(1, n_images-1)], \
            duration=300, loop=0)