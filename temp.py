import pickle as pkl

pkl1 = "dataset/easy/test_10.pkl"
pkl2 = "dataset/easy/expert_episode_test_100_episodes.pkl"

with open(pkl1, 'rb') as f:
    data1 = pkl.load(f)

with open(pkl2, 'rb') as f:
    data2 = pkl.load(f)

replace_key = [35, 63, 82]
idx = 0
for key in data1.keys():
    if idx == len(replace_key):
        break
    data = data1[key]
    data2[replace_key[idx]] = data
    idx += 1

with open("dataset/easy/test_100_episodes.pkl", 'wb') as f:
    pkl.dump(data2, f)

