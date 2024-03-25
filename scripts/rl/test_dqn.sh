# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy_med/algo/dqn_test.yaml
python3 main.py --config $CONFIG \
    --exp easy_med_dqn1 \
    --log_dir log_rl \
    --use_gym