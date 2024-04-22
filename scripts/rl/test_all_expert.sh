
for mode in easy medium hard expert
do
python3 main.py --use_gym --config config/tasks/Nav/${mode}/experts/expert_test.yaml --exp ${mode}_test_normal

python3 main.py --use_gym --config config/tasks/Nav/${mode}/experts/expert_test_slow.yaml --exp ${mode}_test_slow

python3 main.py --use_gym --config config/tasks/Nav/${mode}/experts/expert_test_fast.yaml --exp ${mode}_test_fast
done

# python3 main.py --config config/tasks/Nav/${MODE}/algo/random_test.yaml \
#     --exp random_${MODE}_test \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/${MODE}/experts/expert_test.yaml \
#     --exp oracle_${MODE}_test2 \
#     --log_dir log_rl \
#     --use_gym --save_steps

# python3 main.py --config config/tasks/Nav/${MODE}/experts/expert_val.yaml \
#     --exp oracle_${MODE}_val \
#     --log_dir log_rl \
#     --use_gym --save_steps

# for n in 100
# do
# python3 main.py --config config/tasks/Nav/${MODE}/algo/bctest.yaml \
#     --exp bc_${n}_${MODE} \
#     --checkpoint_path checkpoints/final_models/${MODE}/bc${n}.zip \
#     --log_dir log_rl \
#     --use_gym
# done

# for n in 100
# do
# python3 main.py --config config/tasks/Nav/${MODE}/algo/nlm_test.yaml \
#     --exp nlmbc_${n}_${MODE} \
#     --checkpoint_path checkpoints/final_models/${MODE}/nlmbc_${n}.pth \
#     --log_dir log_rl \
#     --use_gym
# done

# python3 main.py --config config/tasks/Nav/${MODE}/algo/dqn_test.yaml \
#     --exp dqn_${MODE} \
#     --checkpoint_path checkpoints/final_models/${MODE}/dqn.zip \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/${MODE}/algo/a2c_test.yaml \
#     --exp a2c_${MODE} \
#     --checkpoint_path checkpoints/final_models/${MODE}/a2c.zip \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/${MODE}/algo/ppo_test.yaml \
#     --exp ppo_${MODE} \
#     --checkpoint_path checkpoints/final_models/${MODE}/ppo.zip \
#     --log_dir log_rl \
#     --use_gym

# for c in 1 2 3 4 5
# do
# python3 main.py --config config/tasks/Nav/${MODE}/algo/nlmval.yaml \
#     --exp nlmbc_${c}_${MODE}_val \
#     --checkpoint_path checkpoints/logicity_${MODE}100/checkpoints/checkpoint_${c}.pth \
#     --log_dir log_rl \
#     --use_gym
# done

# python3 main.py --config config/tasks/Nav/${MODE}/algo/hri_test.yaml \
#     --exp hri_${MODE} \
#     --log_dir log_rl \
#     --use_gym

# for n in 50 100
# do
# python3 main.py --config config/tasks/Nav/${MODE}/algo/nlm_test.yaml \
#     --exp nlmbc_${n}_${MODE} \
#     --checkpoint_path checkpoints/final_models/hard/nlmbc_${n}.pth \
#     --log_dir log_rl \
#     --use_gym
# done