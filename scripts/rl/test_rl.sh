# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/RL/config_mlp.yaml
python3 main.py --rl_config $CONFIG --exp mlp_test --use_gym