# Description: Script to create episodes data for the evaluation
source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
# /bin/bash
# easy
CONFIG_FILE="config/tasks/Nav/transfer/easy/expert_episode_test.yaml"
python3 tools/create_episode.py --config $CONFIG_FILE --exp "easy_test_transfer" \
    --max_episodes 100
CONFIG_FILE="config/tasks/Nav/transfer/easy/expert_episode_val.yaml"
python3 tools/create_episode.py --config $CONFIG_FILE --exp "easy_val_transfer" \
    --max_episodes 40