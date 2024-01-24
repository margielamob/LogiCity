source /root/miniconda3/etc/profile.d/conda.sh
conda activate logicity
EXPNAME="easy_2k"
MAXSETP=2000
for s in 0 1 2 3 4 5 6 7 8 9
do
python3 main.py --agents "config/agents/v0.yaml" --rules "config/rules/Z3/easy/easy_rule_local.yaml" \
        --exp ${EXPNAME}_${s} --max-steps $MAXSETP --seed $s
done