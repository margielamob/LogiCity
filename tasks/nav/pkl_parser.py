import os
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from core.config import *
from tools.pkl2city import PATH_DICT, SCALE, ICON_SIZE_DICT, resize_with_aspect_ratio, gridmap2img_static

np.random.seed(0)

def parse_pkl(data_path, logger, fov=224, training_ratio=0.95, 
              all_parsed='dataset/nav/parsed_pkls/parsed_all_lonelyped_20k.pkl',
              img_path='dataset/nav/images/global_img.png'):
    train_data = {
        'traj': [],
        'centers': [],
        'local_goals': [],
        'final_goals': [],
        'labels': []
    }
    test_data = {
        'traj': [],
        'centers': [],
        'local_goals': [],
        'final_goals': [],
        'labels': []
    }
    logger.info('Parsing data from {}'.format(data_path))
    if not os.path.exists(img_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            obs = data["Time_Obs"]
            agents = data["Static Info"]["Agents"]
        logger.info('Loaded {} steps'.format(len(data['Time_Obs'])))
        logger.info('1. Building global map...')
        # Build global map
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
        static_map = gridmap2img_static(obs[1]["World"].numpy(), icon_dict)
        logger.info('Done!')
    else:
        assert os.path.exists(all_parsed)
        static_map = cv2.imread(img_path)
        logger.info('Done!')
    logger.info('2. Parsing data...')
    if os.path.exists(all_parsed):
        logger.info('Loading parsed data from {}'.format(all_parsed))
        with open(all_parsed, 'rb') as f:
            all_data = pickle.load(f)
        all_traj = all_data['all_traj']
        all_centers = all_data['all_centers']
        all_final_goals = all_data['all_final_goals']
        all_goals = all_data['all_goals']
        all_labels = all_data['all_labels']
        logger.info('Done!')
    else:
        # Parse data
        all_traj = []
        all_centers = []
        all_final_goals = [] # final goal is the goal of the last step, test set only has final goal, which is maped to local FOV as local goal
        all_goals = []
        all_labels = []
        traj_id = 0
        local_step = 0
        # start 1, not 0
        for key in agents.keys():
            logger.info("Processing agent {}".format(key))
            agent_info = agents[key]
            layer_id = agent_info["id"]
            for key in tqdm(obs.keys()):
                grid = obs[key]["World"].numpy()
                agent_layer = grid[layer_id]
                resized_grid = np.repeat(np.repeat(agent_layer, SCALE, axis=0), SCALE, axis=1)
                type_label = TYPE_MAP[agent_info["type"]]
                # new traj?
                walked = np.where(resized_grid == np.float32(type_label + AGENT_WALKED_PATH_PLUS))
                start = np.where(resized_grid == np.float32(type_label + AGENT_START_PLUS))
                goal = np.where(resized_grid == np.float32(type_label + AGENT_GOAL_PLUS))
                if len(goal[0]) == 0:
                    # no goal, the agent is "AT" the goal, skip
                    continue
                if len(walked[0]) == 0 and len(start[0]) > 0:
                    # new traj
                    traj_id += 1
                    logger.info("New traj: {}".format(traj_id))
                    all_traj.append(traj_id)
                    local_step = 0
                elif len(walked[0]) > 0 and len(start[0]) > 0:
                    local_step += 1
                else:
                    # no start and no walked path, skip
                    continue
                # 1. get agent_loc, defined as the left-top corner of the agent
                agent_loc = np.where(resized_grid == type_label)
                center = np.array([agent_loc[0].min(), agent_loc[1].min()])
                assert resized_grid[center[0], center[1]] == type_label
                all_centers.append({
                    "traj_id": traj_id,
                    "step": local_step,
                    "center": center
                })
                # 2. get final_goal_loc, defined as the left-top corner of the goal
                final_goal_loc = np.where(resized_grid == np.float32(type_label + AGENT_GOAL_PLUS))
                final_goal = np.array([final_goal_loc[0].min(), final_goal_loc[1].min()])
                assert (resized_grid[final_goal[0], final_goal[1]] == np.float32(type_label + AGENT_GOAL_PLUS))
                all_final_goals.append({
                    "traj_id": traj_id,
                    "step": local_step,
                    "goal": final_goal
                })
                # 3. get local_goal_loc, defined as the farthest point in the FOV
                path = np.where(resized_grid == np.float32(type_label + AGENT_GLOBAL_PATH_PLUS))
                max_dis = 0
                if len(path[0])> 0:
                    for i in range(len(path[0])):
                        distance = np.linalg.norm(np.array([path[0][i], path[1][i]]) - center)
                        if (np.abs(path[0][i] - center[0]) > fov//2) or (np.abs(path[1][i] - center[1]) > fov//2):
                            continue
                        elif distance > max_dis:
                            max_dis = distance
                            local_goal = np.array([path[0][i], path[1][i]])
                    assert resized_grid[local_goal[0], local_goal[1]] == np.float32(type_label + AGENT_GLOBAL_PATH_PLUS)
                else:
                    # last step, no path
                    local_goal = final_goal
                    assert resized_grid[local_goal[0], local_goal[1]] == np.float32(type_label + AGENT_GOAL_PLUS)
                all_goals.append({
                    "traj_id": traj_id,
                    "step": local_step,
                    "goal": local_goal
                })
                # 4. get local_action
                all_labels.append({
                    "traj_id": traj_id,
                    "step": local_step,
                    "action": obs[key]['Agent_actions'][layer_id-BASIC_LAYER].numpy()
                })
        # Split train/test
        logger.info('Done!')
        with open(all_parsed, 'wb') as f:
            pickle.dump({
                'all_traj': all_traj,
                'all_centers': all_centers,
                'all_final_goals': all_final_goals,
                'all_goals': all_goals,
                'all_labels': all_labels
            }, f)
    logger.info('3. Splitting train/test...')
    traj_ids = np.unique(all_traj)
    np.random.shuffle(traj_ids)
    train_traj_ids = traj_ids[:int(len(traj_ids) * training_ratio)]
    test_traj_ids = traj_ids[int(len(traj_ids) * training_ratio):]
    for i in range(len(all_centers)):
        traj_id = all_centers[i]["traj_id"]
        if traj_id in train_traj_ids:
            if traj_id not in train_data["traj"]:
                train_data["traj"].append(traj_id)
            train_data["centers"].append(all_centers[i])
            train_data["local_goals"].append(all_goals[i])
            train_data["final_goals"].append(all_final_goals[i])
            train_data["labels"].append(all_labels[i])
            assert all_centers[i]["traj_id"] == all_goals[i]["traj_id"] == all_labels[i]["traj_id"] == all_final_goals[i]["traj_id"]
            assert all_centers[i]["step"] == all_goals[i]["step"] == all_labels[i]["step"] == all_final_goals[i]["step"]
        else:
            assert traj_id in test_traj_ids
            if traj_id not in test_data["traj"]:
                test_data["traj"].append(traj_id)
            test_data["centers"].append(all_centers[i])
            test_data["local_goals"].append(all_goals[i])
            test_data["final_goals"].append(all_final_goals[i])
            test_data["labels"].append(all_labels[i])
            assert all_centers[i]["traj_id"] == all_goals[i]["traj_id"] == all_labels[i]["traj_id"] == all_final_goals[i]["traj_id"]
            assert all_centers[i]["step"] == all_goals[i]["step"] == all_labels[i]["step"] == all_final_goals[i]["step"]
    logger.info('Done!')
    logger.info('Train data size: {} traj, {} steps'.format(len(train_data["traj"]), len(train_data["labels"])))
    logger.info('Test data size: {} traj, {} steps'.format(len(test_data["traj"]), len(test_data["labels"])))
    # logger.info('4. Saving data...')
    # with open(os.path.join(data_dir, "train_data.pkl"), "wb") as f:
    #     pickle.dump(train_data, f) 
    # with open(os.path.join(data_dir, "test_data.pkl"), "wb") as f:
    #     pickle.dump(test_data, f)
    return train_data, test_data, static_map

def visualize_traj(static_map, data, traj_id, fov=224, folder='debug_traj'):
    centers = data["centers"]
    goals = data["goals"]
    labels = data["labels"]
    traj_centers = []
    traj_goals = []
    traj_labels = []
    for i in range(len(centers)):
        if centers[i]["traj_id"] == traj_id:
            traj_centers.append(centers[i])
            traj_goals.append(goals[i])
            traj_labels.append(labels[i])
    traj_centers = sorted(traj_centers, key=lambda x: x["step"])
    traj_goals = sorted(traj_goals, key=lambda x: x["step"])
    traj_labels = sorted(traj_labels, key=lambda x: x["step"])
    for i in range(len(traj_centers)):
        fov_img = np.ones((fov, fov, 3), dtype=np.uint8) * 255
        center = traj_centers[i]["center"]
        goal = traj_goals[i]["goal"]
        label = traj_labels[i]["action"]
        print("step: {}, center: {}, goal: {}, label: {}".format(traj_centers[i]["step"], center, goal, label))
        static_map[center[0]:center[0]+4, center[1]:center[1]+4] = np.array([0, 110, 255])
        static_map[goal[0]:goal[0]+4, goal[1]:goal[1]+4] = np.array([255, 110, 0])
        fov_img = get_fov(static_map, center, fov)
        cv2.imwrite("{}/global_{}.png".format(folder, traj_centers[i]["step"]), static_map)
        cv2.imwrite("{}/fov_{}.png".format(folder, traj_centers[i]["step"]), fov_img)

def get_fov(global_img, center, fov):
    fov_img = np.ones((fov, fov, 3), dtype=np.uint8) * 255
    # Calculate the region of the city image that falls within the FOV
    x_start = max(center[0] - fov//2, 0)
    y_start = max(center[1] - fov//2, 0)
    x_end = min(center[0] + fov//2, global_img.shape[0])
    y_end = min(center[1] + fov//2, global_img.shape[1])

    # Calculate where this region should be placed in the fov_img
    new_x_start = max(fov//2 - center[0], 0)
    new_y_start = max(fov//2 - center[1], 0)
    new_x_end = new_x_start + (x_end - x_start)
    new_y_end = new_y_start + (y_end - y_start)
    fov_img[new_x_start:new_x_end, new_y_start:new_y_end] = global_img[x_start:x_end, y_start:y_end]
    return fov_img

def move(curr_center, action):
    # see agents/pedestrian.py, line 29
    if action == 0:
        return np.array([curr_center[0], curr_center[1] - 1*SCALE])
    elif action == 1:
        return np.array([curr_center[0], curr_center[1] + 1*SCALE])
    elif action == 2:
        return np.array([curr_center[0] - 1*SCALE, curr_center[1]])
    elif action == 3:
        return np.array([curr_center[0] + 1*SCALE, curr_center[1]])
    elif action == 4:
        return np.array([curr_center[0], curr_center[1]])