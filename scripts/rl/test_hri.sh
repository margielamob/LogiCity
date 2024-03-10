# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/medium/algo/hritest_50.yaml
python3 main.py --config $CONFIG \
    --exp med_hri \
    --log_dir log_rl \
    --use_gym