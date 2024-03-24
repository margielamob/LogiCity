# med
<<<<<<< HEAD
CONFIG=config/tasks/Nav/easy/algo/nlm.yaml
for c in 1 2 3 4 5
do
    python3 main.py --config $CONFIG \
        --exp easy_nlm_50_${c} \
        --checkpoint_path checkpoints/final_models/easy/nlm_50/checkpoint_${c}.pth \
=======
CONFIG=config/tasks/Nav/hardium/algo/nlm.yaml
for c in 4 5
do
    python3 main.py --config $CONFIG \
        --exp med_nlm_100_${c} \
        --checkpoint_path checkpoints/final_models/hard/nlm_100/checkpoint_${c}.pth \
>>>>>>> 0dcca990fdab4c896952ed545cf1823ede8bf004
        --log_dir log_rl \
        --use_gym
done
python3 main.py --config $CONFIG \
<<<<<<< HEAD
    --exp easy_nlm_100_5 \
    --checkpoint_path checkpoints/final_models/easy/nlm_100/checkpoint_5.pth \
=======
    --exp med_nlm_50_5 \
    --checkpoint_path checkpoints/final_models/hard/nlm_50/checkpoint_5.pth \
>>>>>>> 0dcca990fdab4c896952ed545cf1823ede8bf004
    --log_dir log_rl \
    --use_gym