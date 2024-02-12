CONFIG=config/tasks/Nav/easy/experts/expert_test.yaml
python3 main.py --config $CONFIG \
    --exp expert_test \
    --log_dir log_rl \
    --use_gym