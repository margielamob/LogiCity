# Description: Collect Expert demonstration
# /bin/bash
CONFIG=config/tasks/Nav/easy/RL/expert_collect.yaml
python3 main.py --config $CONFIG --exp expert_demon_10k --collect_only