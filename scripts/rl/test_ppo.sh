# Description: Test RL agent
# /bin/bash
CONFIG=config/tasks/Nav/easy_med/algo/nlmppo_val.yaml
python3 main.py --config $CONFIG \
    --exp easymed_ppo_val_40000 \
    --checkpoint_path checkpoints/easy_med_pponlm_40000_steps.zip \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/easy_med/algo/nlmppo_test.yaml \
    --exp easymed_ppo_test_40000 \
    --checkpoint_path checkpoints/easy_med_pponlm_40000_steps.zip \
    --log_dir log_rl \
    --use_gym