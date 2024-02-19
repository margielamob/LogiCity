# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

for k in 100 500 1000 2000 5000 10000
do
python3 main.py --config config/tasks/Nav/easy/RL/bc_${k}.yaml \
    --exp easy_bc_${k} \
    --log_dir log_rl \
    --use_gym
done