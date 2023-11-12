'''
Convert cached pkl data to seq data,
For each agent, first split the whole trajectory into several sub-trajectories, each of which denotes one navigation task.
Collect the local FOV images for each sub-trajectory, and save them as a sequence.
Each sub-trajectory will be annotated in a json file, like coco, which contains the following information:
    - img_id: the id of the image
    - img_path: the path to the image
    - task_id: the id of the task, i.e., the sub-trajectory
    - agent_info: concepts and types of the agent of that image
    - current position at that image (x,y) in local coordinate
    - current local goal at that image step (x,y) in local coordinate
    - current action (annotation) at that image, [0, 0, ..., 1, 0, 0, ...], note that the actions for pedestrains should be mapped to the same as cars in 13-dim
Also note that, all the experiments should be done in the vision-based setting, i.e., the agent only has access to the local FOV images, which may be resized.
'''
import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
import pickle as pkl
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

def agent_fov(city_img, agent_center, fov=200):
    # Create a new image with a white background to represent the UAV's field of view
    fov_img = np.ones((fov, fov, 3), dtype=np.uint8) * 255
    clipped = np.clip(agent_center, 0, city_img.shape[0])  # Clipping the values
    agent_center = clipped.astype(int)

    # Calculate the region of the city image that falls within the UAV's field of view
    x_start = max(agent_center[0] - fov//2, 0)
    y_start = max(agent_center[1] - fov//2, 0)
    x_end = min(agent_center[0] + fov//2, city_img.shape[0])
    y_end = min(agent_center[1] + fov//2, city_img.shape[1])

    # Calculate where this region should be placed in the fov_img
    new_x_start = max(fov//2 - agent_center[0], 0)
    new_y_start = max(fov//2 - agent_center[1], 0)
    new_x_end = new_x_start + (x_end - x_start)
    new_y_end = new_y_start + (y_end - y_start)

    # Place the part of the city image that's within the UAV's field of view into the fov_img
    fov_img[new_x_start:new_x_end, new_y_start:new_y_end] = city_img[x_start:x_end, y_start:y_end]

    return fov_img

def main(args):
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

    with open(args.pkl_path, "rb") as f:
        data = pkl.load(f)
        obs = data["Time_Obs"]
        agents = data["Static Info"]["Agents"]
    static_map = gridmap2img_static(obs[1]["World"].numpy(), icon_dict)
    anno_dict = {
        "images": [],
        "annotations": [],
    }

    num_sub_traj = 0
    for key in range(agents.keys()):
        print("Processing agent {}".format(key))
        agent_info = agents[key]
        layer_id = agent_info["id"]
        current_map = static_map.copy()
        for key in tqdm(obs.keys()):
            grid = obs[key]["World"].numpy()
            agent_layer = grid[layer_id]
            resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=1), SCALE, axis=2)
            # 1. Get the agent position using get_pos()
            agent_pos = get_pos(agent_layer)
            # 2. If get_pos() says the agent has reached the goal, num_sub_traj += 1, create a new folder for the next sub_traj
            if agent_pos == goal_pos:
                num_sub_traj += 1
                sub_traj_path = os.path.join(args.seq_path, f"agent_{i}_sub_traj_{num_sub_traj}")
                os.makedirs(sub_traj_path, exist_ok=True)
                continue
            # 3. If get_pos() says the agent has not reached the goal, crop the agent's local FOV image using agent_fov()
            fov_img = agent_fov(current_map, agent_pos, args.fov)
            # 4. Save the cropped image to the corresponding folder
            img_path = os.path.join(sub_traj_path, f"{key}.png")
            cv2.imwrite(img_path, fov_img)
            # 5. Save the annotation to anno_dict
            annotation = {
                "img_id": key,
                "img_path": img_path,
                "task_id": num_sub_traj,
                "agent_info": agent["info"],
                "current_position": agent_pos,
                "current_local_goal": agent["local_goal"],
                "current_action": agent["action"]
            }
            anno_dict["annotations"].append(annotation)
            # 6. Save the image information to anno_dict
            image_info = {
                "file_name": f"{key}.png",
                "height": fov_img.shape[0],
                "width": fov_img.shape[1],
                "id": key
            }
            anno_dict["images"].append(image_info)
    # Outside of the main loop, save the anno_dict to a JSON file
    with open(args.json_path, 'w') as f:
        json.dump(anno_dict, f, indent=4)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='dataset/nav/raw_pkls/lonely_car_20k_0.pkl', help='path to the cached pkl file')
    parser.add_argument('--seq_path', type=str, default='dataset/nav/images', help='path to save the seq data')
    parser.add_argument('--json_path', type=str, default='dataset/nav/annotations', help='path to save the json annotations')
    parser.add_argument('--fov', type=int, default=150, help='the size of the local FOV image')
    parser.add_argument('--colorful', type=str, default=False, help='use the colorful carton-style images or not')
    parser.add_argument('--num_workers', type=int, default=10, help='the number of workers for multiprocessing')
    args = parser.parse_args()
    main(args)