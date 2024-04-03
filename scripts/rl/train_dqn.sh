MODE=easy
CONFIG=config/tasks/Nav/${MODE}/algo/nlmdqn.yaml
python3 main.py --config $CONFIG \
    --exp ${MODE}_nlmdqn_train \
    --log_dir log_rl \
    --use_gym