import torch
from queue import PriorityQueue

class Node:
    def __init__(self, position:torch.tensor, parent:torch.tensor):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic based estimated cost from current node to end
        self.f = 0  # Total cost

    def __eq__(self, other):
        return torch.all(self.position == other.position)

    def __lt__(self, other):
        return self.f < other.f

def heuristic(point_a, point_b):
    # Manhattan distance on a grid
    return torch.abs(point_a[0] - point_b[0]).item() + torch.abs(point_a[1] - point_b[1]).item()

def is_movement_valid(current, next_position, midline_matrix):
    delta_x = next_position[0] - current[0]
    delta_y = next_position[1] - current[1]

    # Using a 4x4 window to determine road orientation
    window_size = 3
    start_x, end_x = max(0, current[0] - window_size), min(midline_matrix.shape[0], current[0] + window_size + 1)
    start_y, end_y = max(0, current[1] - window_size), min(midline_matrix.shape[1], current[1] + window_size + 1)

    local_midline = midline_matrix[start_x:end_x, start_y:end_y]
    vertical_lines = torch.sum(local_midline, dim=1)
    horizontal_lines = torch.sum(local_midline, dim=0)
    flag = True

    if horizontal_lines.max().item()>4:
        assert vertical_lines.max().item()<4
        # vertical line, check delta_y for not cross mid line
        if delta_y < 0 :  # Moving left
            flag = not torch.any(midline_matrix[start_x:end_x, next_position[1]:current[1]])
        elif delta_y > 0: # Moving right
            flag = not torch.any(midline_matrix[start_x:end_x, current[1]:next_position[1]])
        if flag:
        # vertical line, check delta_x for moving on the right
            if delta_x < 0 :  # Moving up
                # mid line on the left
                return torch.any(midline_matrix[start_x:end_x, start_y:current[1]])
            elif delta_x > 0: # Moving down
                # mid line on the right
                return torch.any(midline_matrix[start_x:end_x, current[1]:end_y])
    elif vertical_lines.max().item()>4:
        assert horizontal_lines.max().item()<4
        # horizontal line, check delta_x for not cross mid line
        if delta_x < 0 :  # Moving up
            flag = not torch.any(midline_matrix[next_position[0]:current[0], start_y:end_y])
        elif delta_x > 0: # Moving down
            flag = not torch.any(midline_matrix[current[0]:next_position[0], start_y:end_y])
        if flag:
            # horizontal line, check delta_y for moving on the right
            if delta_y < 0 :  # Moving left
                # mid line on the bottom
                return torch.any(midline_matrix[current[0]:end_x, start_y:end_y])
            elif delta_y > 0: # Moving right
                # mid line on the top
                return torch.any(midline_matrix[start_x:current[0], start_y:end_y])

    return flag
    
def astar_v(movable_map, midline_matrix, start, end):
    start_node = Node(start, None)
    end_node = Node(end, None)

    open_queue = PriorityQueue()
    open_queue.put(start_node)
    open_dict = {start: start_node}
    closed_list = torch.zeros(movable_map.shape, dtype=torch.bool)

    while not open_queue.empty():
        current_node = open_queue.get()
        
        if current_node.position not in open_dict:
            continue
            
        del open_dict[current_node.position]

        closed_list[current_node.position.tolist()[0], current_node.position.tolist()[1]] = True

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        children = []
        # Consider positions 1 and 2 grids away
        for new_position in [torch.tensor((0, -1)), torch.tensor((0, 1)), torch.tensor((-1, 0)), torch.tensor((1, 0)),
                             torch.tensor((0, -2)), torch.tensor((0, 2)), torch.tensor((-2, 0)), torch.tensor((2, 0))]:
            node_position = current_node.position + new_position

            if (node_position[0].item() > (movable_map.shape[0] - 1) or 
                node_position[0].item() < 0 or 
                node_position[1].item() > (movable_map.shape[1] -1) or 
                node_position[1].item() < 0):
                continue

            if not movable_map[node_position.tolist()[0], node_position.tolist()[1]] \
                or closed_list[node_position.tolist()[0], node_position.tolist()[1]]:
                continue

            # Check if the car is moving on the right side of the midline
            if not is_movement_valid(current_node.position, node_position, midline_matrix):
                continue

            new_node = Node(node_position, current_node)
            children.append(new_node)

        for child in children:
            stride = torch.norm(child.position.float() - current_node.position.float(), p=1).item()  # 1 or 2
            child.g = current_node.g + (1 if stride == 2 else 1.5)  # Lower cost for a stride of 2
            child.h = heuristic(child.position, end_node.position)
            child.f = child.g + 5 * child.h

            if child.position in open_dict and child.g > open_dict[child.position].g:
                continue

            open_queue.put(child)
            open_dict[child.position] = child

    return None