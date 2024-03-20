# med
CONFIG=config/tasks/Nav/medium/algo/nlm.yaml
for c in 1 2 3 4 5
do
    python3 main.py --config $CONFIG \
        --exp med_nlm_50_${c} \
        --checkpoint_path checkpoints/final_models/medium/nlm_50/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
    python3 main.py --config $CONFIG \
        --exp med_nlm_100_${c} \
        --checkpoint_path checkpoints/final_models/medium/nlm_100/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done