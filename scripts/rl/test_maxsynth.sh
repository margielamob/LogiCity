# Description: Test RL agent
source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
# /bin/bash
CONFIG=config/tasks/Nav/medium/algo/maxsynthtest.yaml
python3 main.py --config $CONFIG \
    --exp medium_test_maxsynth \
    --use_gym

CONFIG=config/tasks/Nav/hard/algo/maxsynthtest.yaml
python3 main.py --config $CONFIG \
    --exp hard_test_maxsynth \
    --use_gym