MODE=easy

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

# python3 main.py --config config/tasks/Nav/${MODE}/algo/nlmtest.yaml \
#     --exp nlmbc_${MODE}_test \
#     --checkpoint_path checkpoints/final_models/easy/nlm100.pth \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/expert/algo/nlmtest.yaml \
#     --exp nlmbc_expert_test \
#     --checkpoint_path checkpoints/final_models/expert/nlm100.pth \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/${MODE}/algo/dqn_test.yaml \
#     --exp dqn_${MODE} \
#     --checkpoint_path checkpoints/final_models/hard/dqn_140k.zip \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/${MODE}/algo/a2c_test.yaml \
#     --exp a2c_${MODE} \
#     --checkpoint_path checkpoints/final_models/${MODE}/a2c_140k.zip \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/${MODE}/algo/ppo_test.yaml \
#     --exp ppo_${MODE} \
#     --checkpoint_path checkpoints/final_models/hard/ppo_60k.zip \
#     --log_dir log_rl \
#     --use_gym

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

# for iter in 120000 140000
# do
# python3 main.py --use_gym --config config/tasks/Nav/easy/algo/a2ctest.yaml --exp easy_a2c_test_${iter} \
#     --checkpoint_path checkpoints/easy_a2c_${iter}_steps.zip
# done

# for iter in 60000 70000 140000 160000
# do
# python3 main.py --config config/tasks/Nav/transfer/easy/algo/nlmdqn_test.yaml --exp transfer_easy_nlmdqn_test_${iter} \
#     --checkpoint_path checkpoints/transfer_nlmdqn_easy_initial_${iter}_steps --use_gym
# done

# python3 main.py --config config/tasks/Nav/medium/algo/dreamertest.yaml --exp dreamer_medium_test --use_gym

python3 main.py --config config/tasks/Nav/expert/algo/dreamer5_test.yaml --exp dreamer_expert_test --use_gym
