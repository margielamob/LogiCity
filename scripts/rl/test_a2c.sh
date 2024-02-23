# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy/RL/a2ctest.yaml
python3 main.py --config $CONFIG \
    --exp easy_a2c \
    --log_dir log_rl \
    --use_gym