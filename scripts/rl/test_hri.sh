# Description: Test RL agent
# /bin/bash
python3 main.py --config config/tasks/Nav/easy/algo/hritest.yaml \
    --exp easy_hri \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/easy_med/algo/hritest.yaml \
    --exp easymed_hri \
    --log_dir log_rl \
    --use_gym