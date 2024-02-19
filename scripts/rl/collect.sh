# Description: Collect Expert demonstration
# /bin/bash
CONFIG=config/tasks/Nav/easy/experts/expert_collect_train.yaml
python3 main.py --config $CONFIG --exp expert_demon_2k_0219 \
    --log_dir log_rl --collect_only