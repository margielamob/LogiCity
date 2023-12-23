from .lnn import LNNPlanner
from .z3 import Z3Planner

LPlanner_mapper = {
    'LNN': LNNPlanner,
    'Z3': Z3Planner
}