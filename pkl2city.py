import cv2
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
from core.config import *

IMAGE_BASE_PATH = "./imgs"
SCALE = 4

PATH_DICT = {
    "Car": os.path.join(IMAGE_BASE_PATH, "car.jpg"),
    "Pedestrian": os.path.join(IMAGE_BASE_PATH, "pedestrian.png"),
    "Walking Street": os.path.join(IMAGE_BASE_PATH, "walking.png"),
    "Traffic Street": os.path.join(IMAGE_BASE_PATH, "traffic.png"),
    "Overlap": os.path.join(IMAGE_BASE_PATH, "crossing.png"),
    "Gas Station": os.path.join(IMAGE_BASE_PATH, "gas.jpg"),
    "Garage": os.path.join(IMAGE_BASE_PATH, "garage.jpg"),
    "House": [os.path.join(IMAGE_BASE_PATH, "house{}.png").format(i) for i in range(1, 4)],
    "Office": [os.path.join(IMAGE_BASE_PATH, "office{}.png").format(i) for i in range(1, 4)],
    "Store": [os.path.join(IMAGE_BASE_PATH, "store{}.png").format(i) for i in range(1, 4)],
}

ICON_SIZE_DICT = {
    "Car": SCALE*4,
    "Pedestrian": SCALE*2,
    "Walking Street": SCALE*10,
    "Traffic Street": SCALE*10,
    "Overlap": SCALE*10,
    "Gas Station": SCALE*BUILDING_SIZE,
    "Garage": SCALE*BUILDING_SIZE,
    "House": SCALE*BUILDING_SIZE,
    "Office": SCALE*BUILDING_SIZE,
    "Store": SCALE*BUILDING_SIZE,
}

def resize_with_aspect_ratio(image, base_size):
    # Determine the shorter side of the image
    short_side = min(image.shape[:2])
    
    # Calculate the scaling factor
    scale_factor = base_size / short_side
    
    # Calculate the new dimensions of the image
    new_dims = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    
    # Resize the image with the new dimensions
    resized_img = cv2.resize(image, new_dims, interpolation=cv2.INTER_LINEAR)
    
    return resized_img

def gridmap2img_static(gridmap, icon_dict):
    # step 1: get the size of the gridmap, create a blank image with size*SCALE
    gridmap = gridmap.numpy()
    height, width = gridmap.shape[1], gridmap.shape[2]
    img = np.ones((height*SCALE, width*SCALE, 3), np.uint8) * 255  # assuming white background
    resized_grid = np.repeat(np.repeat(gridmap, SCALE, axis=1), SCALE, axis=2)
    
    # step 2: fill the image with walking street icons
    walking_icon = icon_dict["Walking Street"]
    for i in range(0, img.shape[0], walking_icon.shape[0]):
        for j in range(0, img.shape[1], walking_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(walking_icon.shape[0], img.shape[0] - i)
            w_space_left = min(walking_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            img[i:i+h_space_left, j:j+w_space_left] = walking_icon[:h_space_left, :w_space_left]

    # step 3: read the STREET layer of gridmap, paste the traffic street icons on the traffic street region
    # For the traffic icon
    traffic_icon = icon_dict["Traffic Street"]
    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Traffic Street"]
    for i in range(0, img.shape[0], traffic_icon.shape[0]):
        for j in range(0, img.shape[1], traffic_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(traffic_icon.shape[0], img.shape[0] - i)
            w_space_left = min(traffic_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = traffic_icon[:h_space_left, :w_space_left]
    img[traffic_mask] = traffic_img[traffic_mask]

    # For the Overlap and mid lane icon
    traffic_icon = icon_dict["Overlap"]
    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Overlap"]
    for i in range(0, img.shape[0], traffic_icon.shape[0]):
        for j in range(0, img.shape[1], traffic_icon.shape[1]):
            # Calculate the dimensions of the region left in img
            h_space_left = min(traffic_icon.shape[0], img.shape[0] - i)
            w_space_left = min(traffic_icon.shape[1], img.shape[1] - j)

            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = traffic_icon[:h_space_left, :w_space_left]
    img[traffic_mask] = traffic_img[traffic_mask]

    traffic_img = np.zeros_like(img)
    traffic_mask = resized_grid[STREET_ID] == TYPE_MAP["Mid Lane"]
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # Paste the walking_icon (or its sliced version) to img
            traffic_img[i:i+h_space_left, j:j+w_space_left] = [0, 215, 255]
    img[traffic_mask] = traffic_img[traffic_mask]

    # For the building icons
    # for i in range(0, height):
    #     for j in range(0, width):
    #         if gridmap[BUILDING_ID, i, j] != 0:
    #             entity = LABEL_MAP[int(gridmap[BUILDING_ID, i, j].item())] 
    #             if entity in ["House", "Office", "Store"]:
    #                 icon_id = np.random.choice(3)
    #                 icon = icon_dict[entity][icon_id]
    #             else:
    #                 icon = icon_dict[entity]
    #             h_space_left = min(icon.shape[0], img.shape[0] - i*SCALE)
    #             w_space_left = min(icon.shape[1], img.shape[1] - j*SCALE)
    #             img[i*SCALE:i*SCALE+h_space_left, j*SCALE:j*SCALE+w_space_left] = icon[:h_space_left, :w_space_left]

    return img

def gridmap2img_agents(gridmap, icon_dict, static_map):
    return  

def main():
    icon_dict = {}
    for key in PATH_DICT.keys():
        if isinstance(PATH_DICT[key], list):
            raw_img = [cv2.imread(path) for path in PATH_DICT[key]]
            resized_img = [resize_with_aspect_ratio(img, ICON_SIZE_DICT[key]) for img in raw_img]
            icon_dict[key] = resized_img
        else:
            raw_img = cv2.imread(PATH_DICT[key])
            resized_img = resize_with_aspect_ratio(raw_img, ICON_SIZE_DICT[key])
            icon_dict[key] = resized_img

    with open("log/debug.pkl", "rb") as f:
        data = pkl.load(f)
        static_map = gridmap2img_static(data[0], icon_dict)
        for key in tqdm(data.keys()):
            grid = data[key]
            img = gridmap2img_agents(grid, icon_dict, static_map)
            cv2.imwrite("vis_city/{}.png".format(key), img)
        cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()