for n in 50 100
do
CONFIG=config/tasks/Nav/medium/algo/bc_${n}_test.yaml
python3 main.py --config $CONFIG \
    --exp bc_${n}_test \
    --log_dir log_rl \
    --use_gym
done

python3 main.py --config config/tasks/Nav/medium/algo/dqn_test.yaml \
    --exp dqn_test \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/medium/algo/ppo_test.yaml \
    --exp ppo_test \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/medium/algo/a2c_test.yaml \
    --exp a2c_test \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/medium/algo/hritest_50.yaml \
    --exp hri_test \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/medium/algo/random_test.yaml \
    --exp random_test \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/medium/experts/expert_test.yaml \
    --exp oracle_test \
    --log_dir log_rl \
    --use_gym