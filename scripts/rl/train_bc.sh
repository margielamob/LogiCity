# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

for k in 10 50 100 200 500 800 1000
do
python3 main.py --config config/tasks/Nav/easy/RL/bc_${k}.yaml \
    --exp easy_bc_${k} \
    --log_dir log_rl \
    --use_gym
done