source /opt/conda/etc/profile.d/conda.sh
conda activate logicity
EXPNAME="med_1k"
MAXSETP=1000

for s in 0 1 2 3 4
do
python3 main.py --config "config/tasks/sim/med.yaml" \
        --exp ${EXPNAME}_${s} --max-steps $MAXSETP --seed $s \
        --log_dir log_sim
done