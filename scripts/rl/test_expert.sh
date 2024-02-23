# CONFIG=config/tasks/Nav/easy/experts/expert_val.yaml
# python3 main.py --config $CONFIG \
#     --exp expert_val \
#     --log_dir log_rl \
#     --use_gym

CONFIG=config/tasks/Nav/easy/experts/expert_test.yaml
python3 main.py --config $CONFIG \
    --exp expert_test \
    --log_dir log_rl \
    --use_gym