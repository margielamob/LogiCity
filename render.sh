for i in 92 93
do
    echo "Rendering $i"
    python tools/pkl2city.py \
        --pkl log_rl/dqn_hard_test_3_$i.pkl \
        --output_folder vis/dqn/$i
    python tools/pkl2city.py \
        --pkl log_rl/nlmdqn_hard_test_3_$i.pkl \
        --output_folder vis/nlmdqn/$i
done