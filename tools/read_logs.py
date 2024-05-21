import os
import re

log_dir = 'log_rl/tl_nlm_val1'
checkpoints_template = 'checkpoints/logicity_transfer_medium1_tl{data}/checkpoints/checkpoint_{checkpoint}.pth'
data_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
checkpoints = [1, 2, 3, 4, 5]

# Dictionary to store Mean Decision Succ for each data value and checkpoint
results = {data: {} for data in data_values}

# Regular expression patterns to match the relevant lines in the log files
mean_decision_succ_pattern = re.compile(r'Mean Decision Succ: (\d*\.\d+)')

for data in data_values:
    for checkpoint in checkpoints:
        log_filename = f'transfer_easy2med_nlm_val_{checkpoint}_1_{data}.log'
        log_path = os.path.join(log_dir, log_filename)

        with open(log_path, 'r') as log_file:
            for line in log_file:
                mean_decision_succ_match = mean_decision_succ_pattern.search(line)
                if mean_decision_succ_match:
                    mean_decision_succ = float(mean_decision_succ_match.group(1))
                    results[data][checkpoint] = mean_decision_succ
                    break

# Generate the list of selected checkpoints
selected_checkpoints = []
for data, checkpoints_scores in results.items():
    sorted_checkpoints = sorted(checkpoints_scores, key=checkpoints_scores.get, reverse=True)[:2]
    for checkpoint in sorted_checkpoints:
        checkpoint_path = checkpoints_template.format(data=data, checkpoint=checkpoint)
        selected_checkpoints.append(checkpoint_path)

# Write the shell script to a file
with open('scripts/rl/test_all_nlm1.sh', 'w') as f:
    f.write('checkpoints=(\n')
    for checkpoint in selected_checkpoints:
        f.write(f'"{checkpoint}"\n')
    f.write(')\n\n')

    f.write('for c in "${checkpoints[@]}"\n')
    f.write('do\n')
    f.write('  iter=$(echo $c | grep -o "checkpoint_[0-9]*" | cut -d"_" -f2)\n')
    f.write('  data=$(echo $c | grep -o "medium0_tl[0-9]*" | grep -o "[0-9]*$")\n')
    f.write('  python3 main.py --config config/tasks/Nav/transfer/medium/algo/nlmtest.yaml --exp transfer_easy2med_nlm_val_${iter}_0_${data} --checkpoint_path $c --use_gym\n')
    f.write('done\n')
