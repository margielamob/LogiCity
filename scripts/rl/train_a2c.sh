CONFIG=config/tasks/Nav/easy_med/algo/a2c.yaml
python3 main.py --config $CONFIG \
    --exp easy_med_a2c \
    --log_dir log_rl \
    --use_gym