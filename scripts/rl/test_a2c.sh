# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/medium/algo/a2c_test.yaml
python3 main.py --config $CONFIG \
    --exp med_a2c_test \
    --log_dir log_rl \
    --use_gym