# easy
for n in 50 100
    do
    for c in 1 2 3 4 5
    do
        CONFIG=config/tasks/Nav/medium/algo/nlm_${n}_${c}.yaml
        python3 main.py --config $CONFIG \
            --exp med_nlm_${n}_${c} \
            --log_dir log_rl \
            --use_gym
    done
done