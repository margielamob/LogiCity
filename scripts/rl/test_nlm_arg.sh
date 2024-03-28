# med
CONFIG=config/tasks/Nav/easy_med/algo/nlm.yaml
for c in 1 2 3 4 5
do
    python3 main.py --config $CONFIG \
        --exp easymed_nlm_100_val_${c} \
        --checkpoint_path checkpoints/nlm_easy_med_100/checkpoints/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done
for c in 1 2 3 4 5
do
    python3 main.py --config $CONFIG \
        --exp easymed_nlm_50_val_${c} \
        --checkpoint_path checkpoints/nlm_easy_med_50/checkpoints/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done