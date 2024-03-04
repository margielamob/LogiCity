CONFIG=config/tasks/Nav/medium/algo/dqn.yaml
python3 main.py --config $CONFIG \
    --exp medium_dqn \
    --log_dir log_rl \
    --use_gym