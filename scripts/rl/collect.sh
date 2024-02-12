# Description: Collect Expert demonstration
# /bin/bash
CONFIG=config/tasks/Nav/easy/experts/expert_collect.yaml
python3 main.py --config $CONFIG --exp expert_demon_10k \
    --log_dir log_rl --collect_only