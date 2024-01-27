source /root/miniconda3/etc/profile.d/conda.sh
conda activate logicity
python3 main.py --config config/tasks/Nav/RL/config_mlp.yaml --exp PPO_small_region --use_gym