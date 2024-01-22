source /root/miniconda3/etc/profile.d/conda.sh
conda activate logicity
EXP_NAME="easy_occ8_2k"
MAX_SETP=2000

python3 main.py --agents "config/agents/v0.yaml" --rules "config/rules/Z3/easy/easy_rule_local.yaml" \
    --exp $EXP_NAME --max-steps $MAX_SETP --seed 0 --vis False