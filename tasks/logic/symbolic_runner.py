import numpy as np
from tasks.logic.symbolic import DecisionTreeRunner
from tasks.logic.pkl_parser import parse_pkl

def runner(args, logger, writer):
    # Load your dataset
    _, _, data_X, data_Y, _, Yname = parse_pkl(args.data_path, logger)

    dt_runner = DecisionTreeRunner(data_X, data_Y, Yname, logger)
    dt_runner.run()