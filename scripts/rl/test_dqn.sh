# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/hard/algo/dqn_test.yaml
python3 main.py --config $CONFIG \
    --exp hard_dqn1 \
    --log_dir log_rl \
    --use_gym