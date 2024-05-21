checkpoints=(
"checkpoints/logicity_transfer_medium0_tl10/checkpoints/checkpoint_1.pth"
"checkpoints/logicity_transfer_medium0_tl10/checkpoints/checkpoint_2.pth"
"checkpoints/logicity_transfer_medium0_tl20/checkpoints/checkpoint_2.pth"
"checkpoints/logicity_transfer_medium0_tl20/checkpoints/checkpoint_3.pth"
"checkpoints/logicity_transfer_medium0_tl30/checkpoints/checkpoint_5.pth"
"checkpoints/logicity_transfer_medium0_tl30/checkpoints/checkpoint_2.pth"
"checkpoints/logicity_transfer_medium0_tl40/checkpoints/checkpoint_4.pth"
"checkpoints/logicity_transfer_medium0_tl40/checkpoints/checkpoint_5.pth"
"checkpoints/logicity_transfer_medium0_tl50/checkpoints/checkpoint_3.pth"
"checkpoints/logicity_transfer_medium0_tl50/checkpoints/checkpoint_1.pth"
"checkpoints/logicity_transfer_medium0_tl60/checkpoints/checkpoint_1.pth"
"checkpoints/logicity_transfer_medium0_tl60/checkpoints/checkpoint_2.pth"
"checkpoints/logicity_transfer_medium0_tl70/checkpoints/checkpoint_4.pth"
"checkpoints/logicity_transfer_medium0_tl70/checkpoints/checkpoint_1.pth"
"checkpoints/logicity_transfer_medium0_tl80/checkpoints/checkpoint_2.pth"
"checkpoints/logicity_transfer_medium0_tl80/checkpoints/checkpoint_3.pth"
"checkpoints/logicity_transfer_medium0_tl90/checkpoints/checkpoint_4.pth"
"checkpoints/logicity_transfer_medium0_tl90/checkpoints/checkpoint_5.pth"
"checkpoints/logicity_transfer_medium0_tl100/checkpoints/checkpoint_1.pth"
"checkpoints/logicity_transfer_medium0_tl100/checkpoints/checkpoint_4.pth"
)

for c in "${checkpoints[@]}"
do
  iter=$(echo $c | grep -o "checkpoint_[0-9]*" | cut -d"_" -f2)
  data=$(echo $c | grep -o "medium0_tl[0-9]*" | grep -o "[0-9]*$")
  python3 main.py --config config/tasks/Nav/transfer/medium/algo/nlmtest.yaml --exp transfer_easy2med_nlm_val_${iter}_0_${data} --checkpoint_path $c --use_gym
done
