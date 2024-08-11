import numpy as np
import matplotlib.pyplot as plt

# Load data from npy files
mode = 'hard'
rewards1 = np.load('log_rl/oracle_hard_test_3_rewards.npy')
rewards2 = np.load('log_rl/nlmdqn_hard_test_3_rewards.npy')
rewards3 = np.load('log_rl/dqn_hard_test_3_rewards.npy')

# filtering out rewards < -10
rewards1[rewards1 < -30] = -30
rewards2[rewards2 < -30] = -30
rewards3[rewards3 < -30] = -30

# Number of rows and segments
num_rows = 5
segment_length = len(rewards1) // num_rows  # Assuming all files have the same length

# Create figure with subplots
fig, axes = plt.subplots(num_rows, 1, figsize=(10, 15))

# Define the width of each bar and the offset
bar_width = 0.2
offset = 0.2

# Plot each segment
for i in range(num_rows):
    ax = axes[i]
    # Calculate indices for the segment
    start_idx = i * segment_length
    end_idx = start_idx + segment_length
    # Plot each set of rewards
    indices = np.arange(start_idx, end_idx)
    ax.bar(indices - offset, rewards1[start_idx:end_idx], width=bar_width, label='Oracle')
    ax.bar(indices, rewards2[start_idx:end_idx], width=bar_width, label='NLM-DQN')
    ax.bar(indices + offset, rewards3[start_idx:end_idx], width=bar_width, label='DQN')
    ax.set_title(f'Rewards from {start_idx} to {end_idx}')
    ax.legend()

plt.tight_layout()

# Save the figure
plt.savefig('hard_rewards.png')

# Display the plot
plt.show()