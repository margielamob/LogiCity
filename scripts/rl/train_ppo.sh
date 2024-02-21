CONFIG=config/tasks/Nav/easy/RL/ppo.yaml
python3 main.py --config $CONFIG \
    --exp easy_ppo4 \
    --log_dir log_rl \
    --use_gym