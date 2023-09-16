from .A_star import astar
from .A_star_v import astar_v

GPlanner_mapper = {
    'A*': astar,
    'A*v': astar_v
}