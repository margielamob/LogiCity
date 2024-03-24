CONFIG=config/tasks/Nav/easy_med/algo/dqn.yaml
python3 main.py --config $CONFIG \
    --exp easy_med_dqn \
    --log_dir log_rl \
    --use_gym