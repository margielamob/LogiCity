# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy/RL/dqntest.yaml
python3 main.py --config $CONFIG \
    --exp easy_dqn1 \
    --log_dir log_rl \
    --use_gym