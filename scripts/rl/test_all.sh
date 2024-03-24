for n in 100
do
CONFIG=config/tasks/Nav/hardium/algo/bc_${n}_test.yaml
python3 main.py --config $CONFIG \
    --exp bc_${n}_test2 \
    --log_dir log_rl \
    --use_gym
done

# python3 main.py --config config/tasks/Nav/hardium/algo/dqn_test.yaml \
#     --exp dqn_test \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/hardium/algo/ppo_test.yaml \
#     --exp ppo_test \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/hardium/algo/a2c_test.yaml \
#     --exp a2c_test \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/hardium/algo/hritest_50.yaml \
#     --exp hri_test \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/hardium/algo/random_test.yaml \
#     --exp random_test \
#     --log_dir log_rl \
#     --use_gym

# python3 main.py --config config/tasks/Nav/hardium/experts/expert_test.yaml \
#     --exp oracle_test \
#     --log_dir log_rl \
#     --use_gym