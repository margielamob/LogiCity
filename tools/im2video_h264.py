import cv2
import os
from glob import glob

image_folder = 'vis_city'

image_files = glob(os.path.join(image_folder, "*.png"))
imgs = [os.path.join(image_folder, "{}.png".format(k)) for k in range(500)]

# Assuming all images are the same size, get the dimensions of the first image
img = cv2.imread(imgs[0])
height, width, layers = img.shape


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vis_city_video.avi', fourcc, 30.0, (width,  height))

for image_file in imgs:
    img = cv2.imread(image_file)
    out.write(img)

out.release()
cv2.destroyAllWindows()
