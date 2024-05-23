checkpoints=(
"checkpoints/medium_bc1_80_1500_steps.zip"
"checkpoints/medium_bc1_80_2000_steps.zip"
"checkpoints/medium_bc1_50_1000_steps.zip"
"checkpoints/medium_bc1_50_500_steps.zip"
"checkpoints/medium_bc1_40_3500_steps.zip"
"checkpoints/medium_bc1_40_3000_steps.zip"
"checkpoints/medium_bc1_30_500_steps.zip"
"checkpoints/medium_bc1_30_1000_steps.zip"
"checkpoints/medium_bc1_90_3500_steps.zip"
"checkpoints/medium_bc1_90_1000_steps.zip"
"checkpoints/medium_bc1_10_500_steps.zip"
"checkpoints/medium_bc1_10_2500_steps.zip"
"checkpoints/medium_bc1_100_1000_steps.zip"
"checkpoints/medium_bc1_100_3500_steps.zip"
"checkpoints/medium_bc1_70_1000_steps.zip"
"checkpoints/medium_bc1_70_2000_steps.zip"
"checkpoints/medium_bc1_20_3500_steps.zip"
"checkpoints/medium_bc1_20_4000_steps.zip"
"checkpoints/medium_bc1_60_2000_steps.zip"
"checkpoints/medium_bc1_60_500_steps.zip"
)

for c in "${checkpoints[@]}"
do
  exp_name=$(echo $c | grep -o "medium_bc[0-9]*_[0-9]*" | cut -d"_" -f2,3)
  step=$(echo $c | grep -o "[0-9]*_steps" | cut -d"_" -f1)
  run=$(echo $exp_name | cut -d"_" -f1)
  scale=$(echo $exp_name | cut -d"_" -f2)
  python3 main.py --config config/tasks/Nav/transfer/medium/algo/bc_test.yaml --exp transfer_easy2medium_test_${run}_${scale}_${step} --checkpoint_path $c --use_gym
done
