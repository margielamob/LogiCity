checkpoints=(
"checkpoints/medium_bc_100_2500_steps.zip"
"checkpoints/medium_bc_100_4000_steps.zip"
"checkpoints/medium_bc_60_1500_steps.zip"
"checkpoints/medium_bc_60_1000_steps.zip"
"checkpoints/medium_bc_30_1000_steps.zip"
"checkpoints/medium_bc_30_2000_steps.zip"
"checkpoints/medium_bc_70_4000_steps.zip"
"checkpoints/medium_bc_70_1000_steps.zip"
"checkpoints/medium_bc_80_1500_steps.zip"
"checkpoints/medium_bc_80_500_steps.zip"
"checkpoints/medium_bc_50_1500_steps.zip"
"checkpoints/medium_bc_50_2000_steps.zip"
"checkpoints/medium_bc_20_4000_steps.zip"
"checkpoints/medium_bc_20_2000_steps.zip"
"checkpoints/medium_bc_40_4000_steps.zip"
"checkpoints/medium_bc_40_3500_steps.zip"
"checkpoints/medium_bc_10_1000_steps.zip"
"checkpoints/medium_bc_10_2000_steps.zip"
"checkpoints/medium_bc_90_1000_steps.zip"
"checkpoints/medium_bc_90_500_steps.zip"
)

for c in "${checkpoints[@]}"
do
  exp_name=$(echo $c | grep -o "medium_bc[0-9]*_[0-9]*" | cut -d"_" -f2,3)
  step=$(echo $c | grep -o "[0-9]*_steps" | cut -d"_" -f1)
  run=$(echo $exp_name | cut -d"_" -f1)
  scale=$(echo $exp_name | cut -d"_" -f2)
  python3 main.py --config config/tasks/Nav/transfer/medium/algo/bc_test.yaml --exp transfer_easy2medium_test_${run}_${scale}_${step} --checkpoint_path $c --use_gym
done
