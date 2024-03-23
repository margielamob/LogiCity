CONFIG=config/tasks/Nav/hardium/algo/a2c.yaml
python3 main.py --config $CONFIG \
    --exp med_a2c \
    --log_dir log_rl \
    --use_gym