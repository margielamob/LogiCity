CONFIG=config/tasks/Nav/hardium/algo/ppo.yaml
python3 main.py --config $CONFIG \
    --exp med_ppo \
    --log_dir log_rl \
    --use_gym