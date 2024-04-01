CONFIG=config/tasks/Nav/hard/algo/nlmppo.yaml
python3 main.py --config $CONFIG \
    --exp easymed_nlmppo \
    --log_dir log_rl \
    --use_gym