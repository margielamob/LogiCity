EXPNAME="easy_2k"
MAXSETP=2000
for s in 0 1 2 3 4 5 6 7 8 9
do
python3 main.py --config "config/tasks/sim/easy.yaml" \
        --exp ${EXPNAME}_${s} --max-steps $MAXSETP --seed $s
done