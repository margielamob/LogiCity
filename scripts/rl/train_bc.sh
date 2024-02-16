# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

for k in 5 10 30 50 100
do
python3 main.py --config config/tasks/Nav/easy/RL/bc_${k}.yaml \
    --exp train_bc_${k} \
    --log_dir log_rl \
    --use_gym
done