for n in 10 50 100 200 500 800 1000
do
CONFIG=config/tasks/Nav/easy/RL/bctest_${n}.yaml
python3 main.py --config $CONFIG \
    --exp bc_${n}_test \
    --log_dir log_rl \
    --use_gym
done

python3 main.py --config config/tasks/Nav/easy/RL/dqntest.yaml \
    --exp dqn_test \
    --log_dir log_rl \
    --use_gym

python3 main.py --config config/tasks/Nav/easy/RL/ppotest.yaml \
    --exp ppo_test \
    --log_dir log_rl \
    --use_gym