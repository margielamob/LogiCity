# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/hard/algo/a2c_test.yaml
python3 main.py --config $CONFIG \
    --exp medeasy_a2c_test \
    --log_dir log_rl \
    --use_gym