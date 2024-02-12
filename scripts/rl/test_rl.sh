# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy/RL/ppotest.yaml
python3 main.py --config $CONFIG \
    --exp test_ppo \
    --log_dir log_rl \
    --use_gym