# source /opt/conda/etc/profile.d/conda.sh
# conda activate logicity
for n in 50 100 200 500 800 1000
do
CONFIG=config/tasks/Nav/easy_med/algo/bc${n}_test.yaml
python3 main.py --config $CONFIG \
    --exp test_bc_${n} \
    --log_dir log_rl \
    --use_gym
done

# python3 main.py --config config/tasks/Nav/hardium/experts/expert_test.yaml \
#     --exp test_expert2 \
#     --log_dir log_rl \
#     --use_gym