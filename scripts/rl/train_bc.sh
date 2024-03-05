# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

for k in 10 20 50 100
do
python3 main.py --config config/tasks/Nav/medium/algo/bc_${k}.yaml \
    --exp med_bc_${k} \
    --log_dir log_rl \
    --use_gym
done