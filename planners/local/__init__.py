from .LNN import astar

GPlanner_mapper = {
    'LNN': astar,
    'A*v': astar_v,
    'A*vg': ASTAR_G
}