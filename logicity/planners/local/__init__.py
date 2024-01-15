# from .lnn import LNNPlanner
from .z3_global import Z3PlannerGlobal
from .z3_local import Z3PlannerLocal
from .basic import LocalPlanner

LPlanner_mapper = {
    'Z3_Global': Z3PlannerGlobal,
    'Z3_Local': Z3PlannerLocal,
}