source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
for n in 200 500 800 1000
do
CONFIG=config/tasks/Nav/medium/algo/bc_${n}_test.yaml
python3 main.py --config $CONFIG \
    --exp test_bc_${n} \
    --log_dir log_rl \
    --use_gym
done