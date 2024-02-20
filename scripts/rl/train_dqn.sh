CONFIG=config/tasks/Nav/easy/RL/dqn.yaml
python3 main.py --config $CONFIG \
    --exp easy_dqn \
    --log_dir log_rl \
    --use_gym