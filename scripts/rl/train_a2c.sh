CONFIG=config/tasks/Nav/hard/algo/a2c.yaml
python3 main.py --config $CONFIG \
    --exp hard_a2c \
    --log_dir log_rl \
    --use_gym