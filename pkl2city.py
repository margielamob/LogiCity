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
    "Car": [os.path.join(IMAGE_BASE_PATH, "car{}.png").format(i) for i in range(1, 6)],
    "Pedestrian": [os.path.join(IMAGE_BASE_PATH, "pedestrian{}.png").format(i) for i in range(1, 6)],
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
    "Car": SCALE*6,
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

def get_pos(local_layer):
    local_layer[local_layer==0] += 0.1
    pos_layer = local_layer == local_layer.astype(np.int64)
    pixels = torch.nonzero(torch.tensor(pos_layer.astype(np.float32)))
    rows = pixels[:, 0]
    cols = pixels[:, 1]
    left = torch.min(cols).item()
    right = torch.max(cols).item()
    top = torch.min(rows).item()
    bottom = torch.max(rows).item()
    return (left, top, right, bottom)

def flip_icon(icon, left, left_, top, top_, type, last_icon=None):
    if type == "Car":
        # Cars by defual face the top
        if left_ > left:
            # move right
            icon = cv2.flip(cv2.transpose(icon), 1)
        elif left_ < left:
            # move left
            icon = cv2.transpose(icon)
        elif top_ > top:
            icon = cv2.flip(icon, 0)
        elif top_ < top:
            icon = icon
        else:
            icon = last_icon if last_icon is not None else icon
        return icon
    elif type == "Pedestrian":
        if left_ < left:
            icon_left = cv2.flip(icon, 1)
            return icon_left
        else: 
            return icon
    else:
        return icon

def gridmap2img_agents(gridmap, gridmap_, icon_dict, static_map, last_icons=None, agents=None):
    current_map = static_map.copy()
    agent_layer = gridmap[BASIC_LAYER:]
    resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=1), SCALE, axis=2)
    agent_layer_ = gridmap_[BASIC_LAYER:]
    resized_grid_ = np.repeat(np.repeat(agent_layer_, SCALE, axis=1), SCALE, axis=2)
    icon_dict_local = {}

    for i in range(resized_grid.shape[0]):
        local_layer = resized_grid[i]
        left, top, right, bottom = get_pos(local_layer)
        local_layer_ = resized_grid_[i]
        left_, top_, right_, bottom_ = get_pos(local_layer_)
        
        agent_type = LABEL_MAP[local_layer[top, left].item()]
        agent_name = "{}_{}".format(agent_type, BASIC_LAYER + i)
        if agents != None:
            concepts = agents[agent_name]["concepts"]
            is_ambulance = False
            is_bus = False
            is_tiro = False
            is_mayor = False
            if "tiro" in concepts.keys():
                if concepts["tiro"] == 1.0:
                    is_tiro = True
            if "bus" in concepts.keys():
                if concepts["bus"] == 1.0:
                    is_bus = True
            if "ambulance" in concepts.keys():
                if concepts["ambulance"] == 1.0:
                    is_ambulance = True
            if "mayor" in concepts.keys():
                if concepts["mayor"] == 1.0:
                    is_mayor = True
            if is_ambulance:
                icon = icon_dict[agent_type][3]
            elif is_bus:
                icon = icon_dict[agent_type][4]
            elif is_tiro:
                icon = icon_dict[agent_type][2]
            elif is_mayor:
                icon = icon_dict[agent_type][1]
            else:
                if agent_type == "Pedestrian":
                    icon_list = icon_dict[agent_type].copy()
                    icon_list.pop(1)
                    icon_id = i%len(icon_list)
                    icon = icon_list[icon_id]
                if agent_type == "Car":
                    icon_list = icon_dict[agent_type].copy()
                    icon_list.pop(2)
                    icon_list.pop(2)
                    icon_list.pop(2)
                    icon_id = i%len(icon_list)
                    icon = icon_list[icon_id]
        else:
            icon_list = icon_dict[agent_type]
            icon_id = i%len(icon_list)
            icon = icon_list[icon_id]

        if last_icons is not None:
            last_icon = last_icons["{}_{}".format(agent_type, i)]
            icon = flip_icon(icon, left, left_, top, top_, agent_type, last_icon)
            last_icons["{}_{}".format(agent_type, i)] = icon
        else:
            icon = flip_icon(icon, left, left_, top, top_, agent_type)
            icon_dict_local["{}_{}".format(agent_type, i)] = icon

        top_img = max(0, (top+bottom)//2-icon.shape[0]//2)
        left_img = max(0, (left+right)//2-icon.shape[1]//2)
        bottom_img = min(current_map.shape[0], (top+bottom)//2+icon.shape[0]-icon.shape[0]//2)
        right_img = min(current_map.shape[1], (left+right)//2+icon.shape[1]-icon.shape[1]//2)

        icon = icon[:bottom_img-top_img, :right_img-left_img]
        icon_mask = np.sum(icon > 10, axis=2) > 0

        # Paste the walking_icon (or its sliced version) to img
        current_map[top_img:bottom_img, left_img:right_img][icon_mask] = icon[icon_mask]
    if last_icons is not None:
        return current_map, last_icons
    else:
        return current_map, icon_dict_local

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

    with open("log/easy_1k.pkl", "rb") as f:
        data = pkl.load(f)
        obs = data["Time_Obs"]
        agents = data["Static Info"]["Agents"]

    print(obs.keys())
    static_map = gridmap2img_static(obs[1]["World"].numpy(), icon_dict)
    cv2.imwrite("vis_city/static_layout.png", static_map)
    last_icons = None
    for key in tqdm(obs.keys()):
        grid = obs[key]["World"].numpy()
        grid_ = obs[key+1]["World"].numpy()
        img, last_icons = gridmap2img_agents(grid, grid_, icon_dict, static_map, last_icons, agents)
        cv2.imwrite("vis_city/{}.png".format(key), img)
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()