CONFIG=config/tasks/Nav/easy_med/algo/ppo.yaml
python3 main.py --config $CONFIG \
    --exp easymed_ppo \
    --log_dir log_rl \
    --use_gym