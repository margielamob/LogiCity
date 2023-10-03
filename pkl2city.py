import cv2
import numpy as np
import pickle as pkl
import torch
import os
from tqdm import tqdm
from scipy.ndimage import label
from core.config import *

IMAGE_BASE_PATH = "./imgs"
SCALE = 4

PATH_DICT = {
    "Car": os.path.join(IMAGE_BASE_PATH, "car.png"),
    "Pedestrian": os.path.join(IMAGE_BASE_PATH, "pedestrian.png"),
    "Walking Street": os.path.join(IMAGE_BASE_PATH, "walking.png"),
    "Traffic Street": os.path.join(IMAGE_BASE_PATH, "traffic.png"),
    "Overlap": os.path.join(IMAGE_BASE_PATH, "crossing.png"),
    "Gas Station": os.path.join(IMAGE_BASE_PATH, "gas.png"),
    "Garage": os.path.join(IMAGE_BASE_PATH, "garage.png"),
    "House": [os.path.join(IMAGE_BASE_PATH, "house{}.png").format(i) for i in range(1, 4)],
    "Office": [os.path.join(IMAGE_BASE_PATH, "office{}.png").format(i) for i in range(1, 4)],
    "Store": [os.path.join(IMAGE_BASE_PATH, "store{}.png").format(i) for i in range(1, 4)],
}

ICON_SIZE_DICT = {
    "Car": SCALE*8,
    "Pedestrian": SCALE*4,
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
    for building in BUILDING_TYPES:
        building_map = resized_grid[BUILDING_ID] == TYPE_MAP[building]
        building_icon = icon_dict[building]
        labeled_matrix, num = label(building_map)

        for i in range(1, num+1):
            local = torch.tensor(labeled_matrix == i)
            pixels = torch.nonzero(local.float())
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            left = torch.min(cols).item()
            right = torch.max(cols).item()
            top = torch.min(rows).item()
            bottom = torch.max(rows).item()
            if building in ["House", "Office", "Store"]:
                icon_id = np.random.choice(3)
                icon = building_icon[icon_id]
            else:
                icon = building_icon
            icon_mask = np.sum(icon > 1, axis=2) > 0
            img[bottom-icon.shape[0]:bottom, left:left+icon.shape[1]][icon_mask] = icon[icon_mask]

    return img

def gridmap2img_agents(gridmap, icon_dict, static_map):
    current_map = static_map.copy()
    agent_layer = gridmap[BASIC_LAYER:]
    resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=1), SCALE, axis=2)
    for i in range(resized_grid.shape[0]):
        local_layer = resized_grid[i]
        local_layer[local_layer==0] += 0.1
        pos_layer = local_layer == local_layer.astype(np.int64)
        pixels = torch.nonzero(torch.tensor(pos_layer.astype(np.float32)))
        rows = pixels[:, 0]
        cols = pixels[:, 1]
        left = torch.min(cols).item()
        right = torch.max(cols).item()
        top = torch.min(rows).item()
        bottom = torch.max(rows).item()
        agent_type = LABEL_MAP[local_layer[top, left].item()]
        icon = icon_dict[agent_type]
        icon_mask = np.sum(icon > 10, axis=2) > 0
        current_map[bottom-icon.shape[0]:bottom, left:left+icon.shape[1]][icon_mask] = icon[icon_mask]
    return current_map

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
        cv2.imwrite("vis_city/static_layout.png", static_map)
        for key in tqdm(data.keys()):
            grid = data[key]
            img = gridmap2img_agents(grid, icon_dict, static_map)
            cv2.imwrite("vis_city/{}.png".format(key), img)
        cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()