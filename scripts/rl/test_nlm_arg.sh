# med
CONFIG=config/tasks/Nav/medium/algo/nlm_50_1.yaml
for c in 1 2 4 6 10 14 18 20
do
    python3 main.py --config $CONFIG \
        --exp med_nlm_50_bs4_es200_${c} \
        --checkpoint_path checkpoints/final_models/medium/nlm_50_bs4_es200/checkpoints/checkpoint_${c}.pth \
        --log_dir log_rl \
        --use_gym
done