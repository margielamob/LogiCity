# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity

for k in 50 100 200 500 800 1000
do
python3 main.py --config config/tasks/Nav/hard/algo/bc_${k}.yaml \
    --exp easymed_bc_${k} \
    --log_dir log_rl \
    --use_gym
done