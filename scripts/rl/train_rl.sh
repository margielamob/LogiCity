CONFIG=config/tasks/Nav/easy/RL/ppo.yaml
python3 main.py --config $CONFIG \
    --exp train_ppo_1e-4 \
    --log_dir log_rl \
    --use_gym