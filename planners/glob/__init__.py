from .A_star import astar
from .A_star_v import astar_v
from .A_star_graph import ASTAR_G

GPlanner_mapper = {
    'A*': astar,
    'A*v': astar_v,
    'A*vg': ASTAR_G
}