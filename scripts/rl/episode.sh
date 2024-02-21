# Description: Script to create episodes data for the evaluation
source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
# /bin/bash
# easy
CONFIG_FILE="config/tasks/Nav/easy/experts/expert_episode_test.yaml"
python3 tools/create_episode.py --config $CONFIG_FILE --exp "expert_episode_test_40" \
    --max_episodes 40
CONFIG_FILE="config/tasks/Nav/easy/experts/expert_episode_val.yaml"
python3 tools/create_episode.py --config $CONFIG_FILE --exp "expert_episode_val_20" \
    --max_episodes 20