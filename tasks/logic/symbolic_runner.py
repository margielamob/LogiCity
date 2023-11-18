import numpy as np
from tasks.logic.symbolic import DecisionTreeRunner
from tasks.logic.pkl_parser import parse_pkl

def runner(args, logger, writer):
    # Load your dataset
    _, _, data_X_train, data_Y_train, _, Yname = parse_pkl(args.train_data_path, logger)
    _, _, data_X_test, data_Y_test, _, _ = parse_pkl(args.test_data_path, logger)

    dt_runner = DecisionTreeRunner(data_X_train, data_Y_train, data_X_test, data_Y_test, Yname, logger, args.w_bernoulli)
    dt_runner.run()
    report = dt_runner.evaluate()
    report.to_excel(f"{args.log_dir}/{args.exp}/classification_report.xlsx", index=True)