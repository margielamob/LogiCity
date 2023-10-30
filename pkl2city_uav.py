import cv2
import numpy as np
import pickle as pkl
import torch
import os
from tqdm import tqdm
from scipy.ndimage import label
from core.config import *

IMAGE_BASE_PATH = "./imgs"
UAV_TRAJ_PATH = "uav_path.txt"
SCALE = 4

PATH_DICT = {
    "Car": [os.path.join(IMAGE_BASE_PATH, "car{}.png").format(i) for i in range(1, 5)],
    "Pedestrian": [os.path.join(IMAGE_BASE_PATH, "pedestrian{}.png").format(i) for i in range(1, 6)],
    "Walking Street": os.path.join(IMAGE_BASE_PATH, "walking.png"),
    "Traffic Street": os.path.join(IMAGE_BASE_PATH, "traffic.png"),
    "Overlap": os.path.join(IMAGE_BASE_PATH, "crossing.png"),
    "Gas Station": os.path.join(IMAGE_BASE_PATH, "gas.png"),
    "Garage": os.path.join(IMAGE_BASE_PATH, "garage.png"),
    "House": [os.path.join(IMAGE_BASE_PATH, "house{}.png").format(i) for i in range(1, 4)],
    "Office": [os.path.join(IMAGE_BASE_PATH, "office{}.png").format(i) for i in range(1, 4)],
    "Store": [os.path.join(IMAGE_BASE_PATH, "store{}.png").format(i) for i in range(1, 4)],
    "UAV": os.path.join(IMAGE_BASE_PATH, "uav.png"),
}

ICON_SIZE_DICT = {
    "Car": SCALE*6,
    "UAV": SCALE*8,
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

def gridmap2img_agents(gridmap, gridmap_, icon_dict, static_map, last_icons=None):
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

def uav_fov(city_img, uav_center, uav_icon, fov=200):
    # Create a new image with a white background to represent the UAV's field of view
    fov_img = np.ones((fov, fov, 3), dtype=np.uint8) * 255
    clipped = np.clip(uav_center, 0, city_img.shape[0])  # Clipping the values
    uav_center = clipped.astype(int)

    # Calculate the region of the city image that falls within the UAV's field of view
    x_start = max(uav_center[0] - fov//2, 0)
    y_start = max(uav_center[1] - fov//2, 0)
    x_end = min(uav_center[0] + fov//2, city_img.shape[0])
    y_end = min(uav_center[1] + fov//2, city_img.shape[1])

    # Calculate where this region should be placed in the fov_img
    new_x_start = max(fov//2 - uav_center[0], 0)
    new_y_start = max(fov//2 - uav_center[1], 0)
    new_x_end = new_x_start + (x_end - x_start)
    new_y_end = new_y_start + (y_end - y_start)

    # Place the part of the city image that's within the UAV's field of view into the fov_img
    fov_img[new_x_start:new_x_end, new_y_start:new_y_end] = city_img[x_start:x_end, y_start:y_end]

    # Paste the uav_icon on the original image
    city_img_with_uav = city_img.copy()
    top_img = max(0, (uav_center[0] - uav_icon.shape[0]//2))
    left_img = max(0, (uav_center[1] - uav_icon.shape[1]//2))
    bottom_img = min(city_img_with_uav.shape[0], uav_center[0]+uav_icon.shape[0]-uav_icon.shape[0]//2)
    right_img = min(city_img_with_uav.shape[1], uav_center[1]+uav_icon.shape[1]-uav_icon.shape[1]//2)
    uav_icon = uav_icon[:bottom_img-top_img, :right_img-left_img]
    uav_icon_mask = np.sum(uav_icon > 0, axis=2) > 0
    city_img_with_uav[top_img:bottom_img, left_img:right_img][uav_icon_mask] = uav_icon[uav_icon_mask]

    # Draw a dashed square on the original image showing the UAV FOV
    color = (0, 0, 255)  # Blue color
    thickness = 2
    dash_length = 5
    for i in range(0, fov, dash_length * 2):
        # Top border
        start_point = (max(uav_center[1] - fov//2 + i, 0), max(uav_center[0] - fov//2, 0))
        end_point = (min(uav_center[1] - fov//2 + i + dash_length, city_img.shape[1]-1), max(uav_center[0] - fov//2, 0))
        cv2.line(city_img_with_uav, start_point, end_point, color, thickness)

        # Bottom border
        start_point = (max(uav_center[1] - fov//2 + i, 0), min(uav_center[0] + fov//2, city_img.shape[0]-1))
        end_point = (min(uav_center[1] - fov//2 + i + dash_length, city_img.shape[1]-1), min(uav_center[0] + fov//2, city_img.shape[0]-1))
        cv2.line(city_img_with_uav, start_point, end_point, color, thickness)

        # Left border
        start_point = (max(uav_center[1] - fov//2, 0), max(uav_center[0] - fov//2 + i, 0))
        end_point = (max(uav_center[1] - fov//2, 0), min(uav_center[0] - fov//2 + i + dash_length, city_img.shape[0]-1))
        cv2.line(city_img_with_uav, start_point, end_point, color, thickness)

        # Right border
        start_point = (min(uav_center[1] + fov//2, city_img.shape[1]-1), max(uav_center[0] - fov//2 + i, 0))
        end_point = (min(uav_center[1] + fov//2, city_img.shape[1]-1), min(uav_center[0] - fov//2 + i + dash_length, city_img.shape[0]-1))
        cv2.line(city_img_with_uav, start_point, end_point, color, thickness)

    return fov_img, city_img_with_uav


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = np.array([0.0, 0.0])
        self.previous_error = np.array([0.0, 0.0])

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

def calculate_trajectory(waypoints, max_acceleration=1000, dt=0.01, total_frames=10000):
    new_waypoints = np.zeros_like(waypoints)
    new_waypoints[:, 0] = waypoints[:, 1]
    new_waypoints[:, 1] = waypoints[:, 0]
    waypoints = new_waypoints
    waypoint_intervals = 200  # The UAV gets a new waypoint every 20 frames
    num_waypoints = waypoints.shape[0]

    kp, ki, kd = 2.0, 0.1, 0.1  # PID constants
    pid_controller = PIDController(kp, ki, kd, dt)

    frames = np.arange(total_frames)
    trajectory = np.zeros((total_frames, 2))
    velocities = np.zeros((total_frames, 2))
    accelerations = np.zeros((total_frames, 2))

    # Start at the initial position
    trajectory[0] = np.array([20, 963])

    current_waypoint_index = 0

    for i in frames[1:]:  # start from 1 since the 0th frame is the initial position
        if i % waypoint_intervals == 0 and current_waypoint_index < num_waypoints - 1:
            current_waypoint_index += 1

        error = waypoints[current_waypoint_index] - trajectory[i-1]

        acceleration = pid_controller.compute(error)

        norm = np.linalg.norm(acceleration)
        if norm > max_acceleration:
            acceleration = acceleration / norm * max_acceleration

        accelerations[i] = acceleration
        velocities[i] = velocities[i-1] + acceleration * dt
        trajectory[i] = trajectory[i-1] + velocities[i] * dt

    return trajectory

def main():
    icon_dict = {}
    way_points = np.loadtxt(UAV_TRAJ_PATH)
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
        data = pkl.load(f)["Time_Obs"]
        static_map = gridmap2img_static(data[1]["World"], icon_dict)
        dense_uav_waypoints = calculate_trajectory(way_points, total_frames=(len(data.keys())-1)*10)
        cv2.imwrite("vis_city/static_layout.png", static_map)
        last_icons = None
        for key in tqdm(data.keys()):
            grid = data[key]["World"].numpy()
            grid_ = data[key+1]["World"].numpy()
            img, last_icons = gridmap2img_agents(grid, grid_, icon_dict, static_map, last_icons)
            cropped_img, visual_img = uav_fov(img, dense_uav_waypoints[10*key], icon_dict["UAV"])
            cv2.imwrite("vis_city_uav/{}.png".format(key), visual_img)
            cv2.imwrite("vis_city_uav/{}_uav.png".format(key), cropped_img)
        cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()