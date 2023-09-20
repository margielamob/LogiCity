import torch
import networkx as nx
from utils.find import find_midroad_segments

class ASTAR_G:
    def __init__(self, movable_map, midline_matrix, offset):
        self.road_offset = offset
        self.movable_map = movable_map
        self.midroad_segments = find_midroad_segments(midline_matrix)
        self.build_graph(self.midroad_segments)

    def build_graph(self, road_segments):
        self.G = nx.DiGraph()  # Use a directed graph for one-directional edges
        # Connect each start point to its end point
        self.start_lists = []
        self.end_lists = []
        for segment in road_segments:
            mid_start, mid_end = segment
            if mid_start[0] == mid_end[0]:
                assert mid_end[1] > mid_start[1]
                # horizonal mid line
                bottom_s = mid_start + torch.tensor([self.road_offset, -1])
                bottom_e = mid_end + torch.tensor([self.road_offset, 1])
                self.G.add_edge(tuple(bottom_s.tolist()), tuple(bottom_e.tolist()), weight=torch.dist(bottom_s.float(), bottom_e.float()).item())
                self.start_lists.append(bottom_s.unsqueeze(0))
                self.end_lists.append(bottom_e.unsqueeze(0))
                top_s = mid_end + torch.tensor([-self.road_offset, 1])
                top_e = mid_start + torch.tensor([-self.road_offset, -1])
                self.G.add_edge(tuple(top_s.tolist()), tuple(top_e.tolist()), weight=torch.dist(top_s.float(), top_e.float()).item())
                self.start_lists.append(top_s.unsqueeze(0))
                self.end_lists.append(top_e.unsqueeze(0))
            elif mid_start[1] == mid_end[1]:
                assert mid_end[0] > mid_start[0]
                # vertical mid line
                left_s = mid_start + torch.tensor([-1, -self.road_offset])
                left_e = mid_end + torch.tensor([1, -self.road_offset])
                self.G.add_edge(tuple(left_s.tolist()), tuple(left_e.tolist()), weight=torch.dist(left_s.float(), left_e.float()).item())
                self.start_lists.append(left_s.unsqueeze(0))
                self.end_lists.append(left_e.unsqueeze(0))
                right_s = mid_end + torch.tensor([1, self.road_offset])
                right_e = mid_start + torch.tensor([-1, self.road_offset])
                self.G.add_edge(tuple(right_s.tolist()), tuple(right_e.tolist()), weight=torch.dist(right_s.float(), right_e.float()).item())
                self.start_lists.append(right_s.unsqueeze(0))
                self.end_lists.append(right_e.unsqueeze(0))
        
        # Connect each end point to other start points (this only happens in intersections)
        for end_point in self.end_lists:
            distances = torch.norm(torch.cat(self.start_lists, dim=0).float() - end_point.float(), dim=1)
            near_starts = torch.cat(self.start_lists, dim=0)[distances < 12]
            for starts in near_starts.tolist():
                self.G.add_edge(tuple(end_point.tolist()[0]), tuple(starts), weight=torch.dist(end_point.float(), torch.tensor(starts).float()).item())

    def find_nearest_node(self, point, origin_list = 's'):
        intersection = torch.zeros_like(point)
        next_node_list = self.start_lists if origin_list=='s' else self.end_lists
        judging_list = self.end_lists if origin_list=='s' else self.start_lists
        next_node_list = torch.cat(next_node_list, dim=0)
        judging_list = torch.cat(judging_list, dim=0)
        dis = torch.abs(point - next_node_list)
        min_dis_value, ind = torch.min(dis, dim=0)
        pos_ind = min_dis_value.min(dim=0).indices.item()
        flag = False
        if pos_ind == 1:
            # on vertical street
            intersection[0] = point[0]
            intersection[1] = next_node_list[ind[pos_ind]][pos_ind]
            filtered_node_list = next_node_list[next_node_list[:, pos_ind] == intersection[1]]
            filtered_juding_list = judging_list[judging_list[:, pos_ind] == intersection[1]]
            # find the goal that so not have start in between
            for k in range(filtered_node_list.shape[0]):
                candidate = filtered_node_list[k]
                top = filtered_juding_list[:, 0]<= max(candidate[0].item(), point[0].item())
                bottom = filtered_juding_list[:, 0] >= min(candidate[0].item(), point[0].item())
                if torch.any(top & bottom):
                    continue
                else:
                    flag = True
                    break
        else:
            # on horizonal street
            intersection[1] = point[1]
            intersection[0] = next_node_list[ind[pos_ind]][pos_ind]
            filtered_node_list = next_node_list[next_node_list[:, pos_ind] == intersection[0]]
            filtered_juding_list = judging_list[judging_list[:, pos_ind] == intersection[0]]
            # find the goal that so not have start in between
            for k in range(filtered_node_list.shape[0]):
                candidate = filtered_node_list[k]
                left = filtered_juding_list[:, 1]<= max(candidate[1].item(), point[1].item())
                right = filtered_juding_list[:, 1] >= min(candidate[1].item(), point[1].item())
                if torch.any(left & right):
                    continue
                else:
                    flag = True
                    break
        
        assert flag
        return intersection, candidate

    def a_star_on_graph(G, start, end):
        # Here we're just calling the NetworkX A* but you can replace it with your own A* for more customization.
        path = nx.astar_path(G, start, end, weight='weight')
        return path


    def plan(self, start, end):

        # Find intersections of start and end with their nearest road segments.
        intersect, close_goal = self.find_nearest_node(start, origin_list = 'g')
        self.G.add_edge(tuple(start.tolist()), tuple(intersect.tolist()))
        self.G.add_edge(tuple(intersect.tolist()), tuple(close_goal.tolist()))
        intersect, close_start = self.find_nearest_node(end)
        self.G.add_edge(tuple(close_start.tolist()), tuple(intersect.tolist()))
        self.G.add_edge(tuple(intersect.tolist()), tuple(end.tolist()))

        path_on_graph = nx.astar_path(self.G, tuple(start.tolist()), tuple(end.tolist()))
        interpolated = self.interpolate(path_on_graph)

        # self.G.remove_edge(tuple(start.tolist()), tuple(intersect.tolist()))
        # self.G.remove_edge(tuple(intersect.tolist()), tuple(close_goal.tolist()))
        # self.G.remove_edge(tuple(close_start.tolist()), tuple(intersect.tolist()))
        # self.G.remove_edge(tuple(intersect.tolist()), tuple(end.tolist()))

        # If necessary, interpolate waypoints between the resulting nodes to form a complete path.
        return interpolated

    def interpolate(self, path_on_graph):
        interpolated = []

        for i in range(len(path_on_graph) - 1):
            current_point = path_on_graph[i]
            next_point = path_on_graph[i+1]

            while current_point != next_point:
                interpolated.append(torch.tensor(current_point))

                # Determine the dominant direction (X or Y)
                dx = next_point[0] - current_point[0]
                dy = next_point[1] - current_point[1]

                if abs(dx) > abs(dy):
                    # Move along X axis by 1 or 2 grids depending on the distance
                    step = 2 if abs(dx) >= 2 else 1
                    step *= int(dx/abs(dx)) if dx != 0 else 0  # Get the direction of movement
                    current_point = (current_point[0] + step, current_point[1])
                else:
                    # Move along Y axis by 1 or 2 grids depending on the distance
                    step = 2 if abs(dy) >= 2 else 1
                    step *= int(dy/abs(dy)) if dy != 0 else 0  # Get the direction of movement
                    current_point = (current_point[0], current_point[1] + step)

        interpolated.append(torch.tensor(path_on_graph[-1]))  # Add the last point

        return interpolated