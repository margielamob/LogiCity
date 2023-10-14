import cv2
import numpy as np
import os
from tqdm import tqdm


def combine_images(img1, img2, canvas_size, border=4, interval=50):
    """
    Combine two images on a canvas, aligning them along the bottom.

    :param img1: First image (larger one, e.g., 964x964).
    :param img2: Second image (smaller one, e.g., 200x200).
    :param canvas_size: Tuple, size of the canvas (height, width).
    :param border: Integer, thickness of the border to be added around images.
    :param interval: Integer, distance between the two images.
    :return: Combined image.
    """
    # Create a white canvas
    canvas = np.ones((*canvas_size, 3), dtype=np.uint8) * 255

    # Calculate positions
    y_offset_img1 = canvas_size[0] - img1.shape[0] - 2*border
    x_offset_img1 = border
    y_offset_img2 = canvas_size[0] - img2.shape[0] - 2*border
    x_offset_img2 = img1.shape[1] + interval + border

    # Draw black border around images
    img1 = cv2.copyMakeBorder(img1, border, border, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img2 = cv2.copyMakeBorder(img2, border, border, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Paste images onto the canvas
    canvas[y_offset_img1:y_offset_img1+img1.shape[0], x_offset_img1:x_offset_img1+img1.shape[1]] = img1
    canvas[y_offset_img2:y_offset_img2+img2.shape[0], x_offset_img2:x_offset_img2+img2.shape[1]] = img2

    return canvas

# Parameters
image_folder = 'vis_city_uav'
output_video = 'output_video.avi'
frame_rate = 10

# Load your images
image1 = cv2.imread("vis_city_uav/{}.png".format(1))  # Replace with your actual file path
image2 = cv2.imread("vis_city_uav/{}_uav.png".format(1))  # Replace with your actual file path

# Define canvas size
canvas_height = max(image1.shape[0], image2.shape[0]) + 8  # max height of images + border*2
canvas_width = image1.shape[1] + image2.shape[1] + 50 + 16  # sum of widths + interval + border*2*2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (canvas_width, canvas_height))

for key in tqdm(range(1000)):  # Assuming each key has two images
    image1 = cv2.imread(os.path.join(image_folder, "{}.png".format(key)))
    image2 = cv2.imread(os.path.join(image_folder, "{}_uav.png".format(key)))

    # Process the images
    processed_img = combine_images(image1, image2, (canvas_height, canvas_width))

    # Write the processed frame
    video_writer.write(processed_img)

# Release everything
video_writer.release()
cv2.destroyAllWindows()
