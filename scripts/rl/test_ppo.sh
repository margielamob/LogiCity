# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy/RL/ppotest.yaml
python3 main.py --config $CONFIG \
    --exp easy_ppo3 \
    --log_dir log_rl \
    --use_gym