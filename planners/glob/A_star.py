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
    
def astar(movable_map, start, end):
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
        for new_position in [torch.tensor((0, -1)), torch.tensor((0, 1)), torch.tensor((-1, 0)), torch.tensor((1, 0))]:
            node_position = current_node.position + new_position

            if (node_position[0].item() > (movable_map.shape[0] - 1) or 
                node_position[0].item() < 0 or 
                node_position[1].item() > (movable_map.shape[1] -1) or 
                node_position[1].item() < 0):
                continue

            if not movable_map[node_position.tolist()[0], node_position.tolist()[1]] \
                or closed_list[node_position.tolist()[0], node_position.tolist()[1]]:
                continue

            new_node = Node(node_position, current_node)
            children.append(new_node)

        for child in children:
            child.g = current_node.g + 1
            child.h = heuristic(child.position, end_node.position)
            child.f = child.g + child.h

            if child.position in open_dict and child.g > open_dict[child.position].g:
                continue

            open_queue.put(child)
            open_dict[child.position] = child

    return None
