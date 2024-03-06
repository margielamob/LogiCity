# CONFIG=config/tasks/Nav/easy/experts/expert_val.yaml
# python3 main.py --config $CONFIG \
#     --exp expert_val \
#     --log_dir log_rl \
#     --use_gym

CONFIG=config/tasks/Nav/medium/experts/expert_test.yaml
python3 main.py --config $CONFIG \
    --exp test_expert \
    --log_dir log_rl \
    --use_gym