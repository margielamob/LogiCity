checkpoints=(
"checkpoints/medium_bc2_40_1500_steps.zip"
"checkpoints/medium_bc2_40_500_steps.zip"
"checkpoints/medium_bc2_10_1500_steps.zip"
"checkpoints/medium_bc2_10_2500_steps.zip"
"checkpoints/medium_bc2_90_500_steps.zip"
"checkpoints/medium_bc2_90_1000_steps.zip"
"checkpoints/medium_bc2_50_3000_steps.zip"
"checkpoints/medium_bc2_50_500_steps.zip"
"checkpoints/medium_bc2_80_2000_steps.zip"
"checkpoints/medium_bc2_80_3500_steps.zip"
"checkpoints/medium_bc2_70_1500_steps.zip"
"checkpoints/medium_bc2_70_3000_steps.zip"
"checkpoints/medium_bc2_20_500_steps.zip"
"checkpoints/medium_bc2_20_1000_steps.zip"
"checkpoints/medium_bc2_30_500_steps.zip"
"checkpoints/medium_bc2_30_2000_steps.zip"
"checkpoints/medium_bc2_100_500_steps.zip"
"checkpoints/medium_bc2_100_1500_steps.zip"
"checkpoints/medium_bc2_60_1000_steps.zip"
"checkpoints/medium_bc2_60_2000_steps.zip"
)

for c in "${checkpoints[@]}"
do
  exp_name=$(echo $c | grep -o "medium_bc[0-9]*_[0-9]*" | cut -d"_" -f2,3)
  step=$(echo $c | grep -o "[0-9]*_steps" | cut -d"_" -f1)
  run=$(echo $exp_name | cut -d"_" -f1)
  scale=$(echo $exp_name | cut -d"_" -f2)
  python3 main.py --config config/tasks/Nav/transfer/medium/algo/bc_test.yaml --exp transfer_easy2medium_test_${run}_${scale}_${step} --checkpoint_path $c --use_gym
done
