# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy_med/algo/nlmppo_test.yaml
python3 main.py --config $CONFIG \
    --exp easymed_ppo_test_30000 \
    --checkpoint_path checkpoints/easy_med_pponlm_30000_steps.zip \
    --log_dir log_rl \
    --use_gym

# python3 main.py --config $CONFIG \
#     --exp easymed_ppo_test_120000 \
#     --checkpoint_path checkpoints/easy_med_pponlm_120000_steps.zip \
#     --log_dir log_rl \
#     --use_gym