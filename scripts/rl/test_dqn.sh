# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/medium/algo/dqn_test.yaml
python3 main.py --config $CONFIG \
    --exp med_dqn1 \
    --log_dir log_rl \
    --use_gym