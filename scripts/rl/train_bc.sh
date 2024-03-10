# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

for k in 200 500 800 1000
do
python3 main.py --config config/tasks/Nav/medium/algo/bc_${k}.yaml \
    --exp med_bc_${k} \
    --log_dir log_rl \
    --use_gym
done