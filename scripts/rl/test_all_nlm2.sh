checkpoints=(
"checkpoints/logicity_transfer_medium2_tl50/checkpoints/checkpoint_1.pth"
"checkpoints/logicity_transfer_medium2_tl50/checkpoints/checkpoint_4.pth"
"checkpoints/logicity_transfer_medium2_tl50/checkpoints/checkpoint_5.pth"
)

for c in "${checkpoints[@]}"
do
  iter=$(echo $c | grep -o "checkpoint_[0-9]*" | cut -d"_" -f2)
  data=$(echo $c | grep -o "medium2_tl[0-9]*" | grep -o "[0-9]*$")
  python3 main.py --config config/tasks/Nav/transfer/medium/algo/nlmtest.yaml --exp transfer_easy2med_nlm_test_${iter}_2_${data} --checkpoint_path $c --use_gym
done
