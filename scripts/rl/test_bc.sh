source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
for n in 1 2 5
do
CONFIG=config/tasks/Nav/easy/RL/bctest_${n}k.yaml
python3 main.py --config $CONFIG \
    --exp test_bc_${n}k \
    --log_dir log_rl \
    --use_gym
done

CONFIG=config/tasks/Nav/easy/RL/bctest_500.yaml
python3 main.py --config $CONFIG \
    --exp test_bc_500 \
    --log_dir log_rl \
    --use_gym