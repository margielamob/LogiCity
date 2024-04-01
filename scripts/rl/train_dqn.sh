CONFIG=config/tasks/Nav/hard/algo/dqn.yaml
python3 main.py --config $CONFIG \
    --exp hard_dqn \
    --log_dir log_rl \
    --use_gym