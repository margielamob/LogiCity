import numpy as np
import cv2
import torch

def visualize_city(city, resolution, agent_layer=None, agent_type=None, file_name="city.png"):
    # Define a color for each entity
    # only visualize a static map with buildings, streets, and one layer agents
    color_map = {
        -1: [100, 100, 100],
        0: [200, 200, 200],       # Grey for empty
        1: [152, 216, 170],           # Green for walking street
        2: [168, 161, 150],        # Red for traffic street
        3: [255, 204, 112],       # house
        4: [34, 102, 141],        # gas station
        5: [255, 250, 221],       # office
        6: [142, 205, 221],       # garage
        7: [255, 63, 164]         # store
    }

    # Create a visual grid of the city, buildings and streets
    visual_grid = np.ones((resolution, resolution, 3), dtype=np.uint8)*200
    np_grid = city.city_grid.numpy().astype(np.int)
    scale_factor = resolution/city.grid_size[0]
    for k in range(2):
        for i in range(city.grid_size[0]):
            for j in range(city.grid_size[1]):
                if np_grid[k][i][j] != 0:
                    color = color_map[np_grid[k][i][j]]
                    visual_grid[int(i*scale_factor):int((i+1)*scale_factor), int(j*scale_factor):int((j+1)*scale_factor)] = color

    # draw agent's if provided layer
    if agent_layer != None:
        assert agent_type != None
        visual_grid = vis_agent(visual_grid, city, agent_layer, agent_type, scale_factor)
    
    # Add the legend
    padding = int(10*scale_factor)
    legend_width = resolution
    legend_height = resolution
    legend_item_height = int(20*scale_factor)
    legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # white background

    for idx, (key, value) in enumerate(color_map.items()):
        y_offset = idx * legend_item_height + padding
        if y_offset + legend_item_height > legend_height:  # Ensure we don't render beyond the legend image
            break
        cv2.rectangle(legend_img, (padding, y_offset), (padding + legend_item_height, y_offset + legend_item_height), color_map[key], -1)
        cv2.putText(legend_img, str(city.label2type[key]), (padding + int(30*scale_factor), y_offset + legend_item_height - int(5*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*scale_factor, \
            (0, 0, 0), int(scale_factor), lineType=cv2.LINE_AA)

    # Combine the visual grid and the legend side by side
    combined_img = np.hstack((visual_grid, legend_img))

    # Use OpenCV to display the city
    cv2.imwrite(file_name, combined_img)

def vis_agent(vis_grid, city, agent_layer, agent_type, scale_factor, curr=True, s=True, g=True, path=True):
    agent_layer = city.city_grid[agent_layer]
    cur_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]).float()).tolist()
    planned_traj = torch.nonzero((agent_layer == city.type2label[agent_type]+0.1).float()).tolist()
    walked_traj = torch.nonzero((agent_layer == city.type2label[agent_type]-0.1).float()).tolist()
    goal_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]+0.3).float()).tolist()
    start_agent_pos = torch.nonzero((agent_layer == city.type2label[agent_type]-0.2).float()).tolist()
    # Path
    if len(planned_traj) > 0 and path:  
        for way_point in planned_traj:
            vis_grid[int(way_point[0]*scale_factor):int((way_point[0]+2)*scale_factor), \
                int(way_point[1]*scale_factor):int((way_point[1]+2)*scale_factor)] = [0, 192, 255]
    if len(walked_traj) > 0 and path:  
        for way_point_ in walked_traj:
            vis_grid[int(way_point_[0]*scale_factor):int((way_point_[0]+2)*scale_factor), \
                int(way_point_[1]*scale_factor):int((way_point_[1]+2)*scale_factor)] = [0, 0, 0]
    # Points
    if len(start_agent_pos) > 0 and s:
        cv2.drawMarker(vis_grid, (int(start_agent_pos[0][1]*scale_factor), int(start_agent_pos[0][0]*scale_factor)), \
            (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2, line_type=cv2.LINE_AA)
    if len(goal_agent_pos) > 0 and g:   
        cv2.drawMarker(vis_grid, (int(goal_agent_pos[0][1]*scale_factor), int(goal_agent_pos[0][0]*scale_factor)), \
            (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2, line_type=cv2.LINE_AA)
    assert len(cur_agent_pos) == 1
    if curr:
        cv2.drawMarker(vis_grid, (int(cur_agent_pos[0][1]*scale_factor), int(cur_agent_pos[0][0]*scale_factor)), \
            (255, 0, 0), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=3, line_type=cv2.LINE_AA)

    return vis_grid