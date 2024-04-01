# med
CONFIG=config/tasks/Nav/hard/algo/nlm.yaml
for c in 5
do
    python3 main.py --config $CONFIG \
        --exp easymed_nlm_100_test_${c} \
        --checkpoint_path checkpoints/nlm_hard_100/checkpoints/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done
for c in 4
do
    python3 main.py --config $CONFIG \
        --exp easymed_nlm_50_test_${c} \
        --checkpoint_path checkpoints/nlm_hard_50/checkpoints/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done