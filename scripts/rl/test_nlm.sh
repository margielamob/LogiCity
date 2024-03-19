# easy
CONFIG=config/tasks/Nav/medium/algo/nlm_900.yaml
python3 main.py --config $CONFIG \
    --exp med_nlm_900 \
    --log_dir log_rl \
    --use_gym