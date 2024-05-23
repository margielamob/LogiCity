import os
import re

log_dir = 'checkpoints'
results_pattern = re.compile(
    r'Step: (\d+) - Success Rate: [\d.]+ - Mean Reward: [\d.-]+\s*'
    r'Mean Decision Succ: [\d.]+\s*'
    r'Average Decision Succ: ([\d.]+)',
    re.MULTILINE
)
file_pattern = re.compile(r'transfer_easy2medium_(bc\d*_\d+)_eval_rewards.txt')

# Dictionary to store Average Decision Succ for each experiment and step
results = {}

# Read the files and extract the Average Decision Succ values
for log_filename in os.listdir(log_dir):
    file_match = file_pattern.match(log_filename)
    if file_match:
        exp_name = file_match.group(1)
        log_path = os.path.join(log_dir, log_filename)
        
        if exp_name not in results:
            results[exp_name] = []

        print(f'Reading file: {log_path}')  # Debug print
        with open(log_path, 'r') as log_file:
            log_content = log_file.read()
            matches = results_pattern.findall(log_content)
            if matches:
                for match in matches:
                    step = int(match[0])
                    avg_decision_succ = float(match[1])
                    print(f'Found: exp_name={exp_name}, step={step}, avg_decision_succ={avg_decision_succ}')  # Debug print
                    results[exp_name].append((step, avg_decision_succ))
            else:
                print(f'No matches found in file: {log_path}')  # Debug print

# Select the top 2 models for each experiment
selected_checkpoints = []
for exp_name, scores in results.items():
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
    for step, _ in top_scores:
        checkpoint_path = f'checkpoints/medium_{exp_name}_{step}_steps.zip'
        selected_checkpoints.append(checkpoint_path)
        print(f'Selected checkpoint: {checkpoint_path}')  # Debug print

# Write the shell script to a file
with open('scripts/rl/test_bc_checkpoints.sh', 'w') as f:
    f.write('checkpoints=(\n')
    for checkpoint in selected_checkpoints:
        f.write(f'"{checkpoint}"\n')
    f.write(')\n\n')

    f.write('for c in "${checkpoints[@]}"\n')
    f.write('do\n')
    f.write('  exp_name=$(echo $c | grep -o "medium_bc[0-9]*_[0-9]*" | cut -d"_" -f2,3)\n')
    f.write('  step=$(echo $c | grep -o "[0-9]*_steps" | cut -d"_" -f1)\n')
    f.write('  python3 main.py --config config/tasks/Nav/transfer/medium/algo/bc_test.yaml --exp transfer_easy2medium_${exp_name} --checkpoint_path $c --use_gym\n')
    f.write('done\n')
