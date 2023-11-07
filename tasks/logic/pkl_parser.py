import pickle
import torch
from tqdm import tqdm

def parse_pkl(data_path, logger):
    print('Parsing data from {}'.format(data_path))
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    LNN_predictates = data['Static Info']['Logic']['Predicates']
    all_data = []
    for i in tqdm(range(1, len(data['Time_Obs']) + 1)):
        time_data = data['Time_Obs'][i]["LNN_state"]
        all_data.append(time_data)
    # logic is independent of time or agents
    all_data = torch.cat(all_data, dim=0)
    # state index
    k = 0
    for i in range(len(LNN_predictates)):
        if LNN_predictates[i] != 'Stop' and LNN_predictates[i] != 'Normal' and LNN_predictates[i] != 'Fast' and LNN_predictates[i] != 'Slow':
            k += 1
    Xs = all_data[:, :k]
    Xs_name = LNN_predictates[:k]
    # use averaged lower and upper bound as the input/output
    Xs = Xs.mean(dim=-1, keepdim=False)
    Ys = all_data[:, k:]
    Ys_name = LNN_predictates[k:]
    Ys = Ys.mean(dim=-1, keepdim=False)
    input_sz = Xs.shape[1]
    output_sz = Ys.shape[1]
    logger.info("All logic predicates: {}".format(LNN_predictates))
    logger.info('Input Logic Pred size: {}'.format(input_sz))
    logger.info('Output Logic Pred size: {}'.format(output_sz))
    return input_sz, output_sz, Xs, Ys, Xs_name, Ys_name