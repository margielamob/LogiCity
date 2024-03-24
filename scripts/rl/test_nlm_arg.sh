# med
CONFIG=config/tasks/Nav/easy/algo/nlm.yaml
for c in 1 2 3 4 5
do
    python3 main.py --config $CONFIG \
        --exp easy_nlm_50_${c} \
        --checkpoint_path checkpoints/final_models/easy/nlm_50/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done
python3 main.py --config $CONFIG \
    --exp easy_nlm_100_5 \
    --checkpoint_path checkpoints/final_models/easy/nlm_100/checkpoint_5.pth \
    --log_dir log_rl \
    --use_gym