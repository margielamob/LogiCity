# Description: Collect Expert demonstration
source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
# /bin/bash
CONFIG=config/tasks/Nav/easy/experts/expert_collect_train.yaml
python3 main.py --config $CONFIG --exp expert_demon_1k_0219 \
    --log_dir log_rl --collect_only