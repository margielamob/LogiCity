# easy
CONFIG=config/tasks/Nav/easy/algo/nlm_900.yaml
python3 main.py --config $CONFIG \
    --exp easy_nlm_900 \
    --log_dir log_rl \
    --use_gym