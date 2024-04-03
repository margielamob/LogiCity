MODE=easy
CONFIG=config/tasks/Nav/${MODE}/algo/dqn.yaml
python3 main.py --config $CONFIG \
    --exp ${MODE}_dqn_train \
    --log_dir log_rl \
    --use_gym