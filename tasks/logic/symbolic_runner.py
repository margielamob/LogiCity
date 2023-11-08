import numpy as np
from tasks.logic.symbolic import DecisionTreeRunner
from tasks.logic.pkl_parser import parse_pkl

def runner(args, logger, writer):
    # Load your dataset
    _, _, data_X_train, data_Y_train, _, Yname = parse_pkl(args.train_data_path, logger)
    _, _, data_X_test, data_Y_test, _, _ = parse_pkl(args.test_data_path, logger)

    dt_runner = DecisionTreeRunner(data_X_train, data_Y_train, data_X_test, data_Y_test, Yname, logger)
    dt_runner.run()
    dt_runner.evaluate()