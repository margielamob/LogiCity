# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/RL/config_test.yaml
python3 main.py --config $CONFIG --exp ppo_small --use_gym