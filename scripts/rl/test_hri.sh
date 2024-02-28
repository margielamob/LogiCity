# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy/RL/hritest_50.yaml
python3 main.py --config $CONFIG \
    --exp easy_hri \
    --log_dir log_rl \
    --use_gym