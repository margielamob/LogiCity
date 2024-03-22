# med
CONFIG=config/tasks/Nav/medium/algo/nlm.yaml
for c in 4 5
do
    python3 main.py --config $CONFIG \
        --exp med_nlm_100_${c} \
        --checkpoint_path checkpoints/final_models/medium/nlm_100/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done
python3 main.py --config $CONFIG \
    --exp med_nlm_50_5 \
    --checkpoint_path checkpoints/final_models/medium/nlm_50/checkpoint_5.pth \
    --log_dir log_rl \
    --use_gym