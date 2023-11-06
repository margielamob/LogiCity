import pickle

def parse_pkl(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data