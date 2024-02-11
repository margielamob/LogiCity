# Description: Collect Expert demonstration
# /bin/bash
CONFIG=config/tasks/Nav/easy/experts/expert_collect.yaml
python3 main.py --config $CONFIG --exp expert_demon_5 \
    --log_dir log_rl --collect_only