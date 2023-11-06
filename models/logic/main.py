import torch
import argparse
import tensorboardX


def main(args):
    # Load model
    model = torch.load(args.model_path)
    # Load data
    data = torch.load(args.data_path)
    # Run model
    model(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='data/data.pt')
    parser.add_argument('--log_dir', type=str, default='logs/logic')

    args = parser.parse_args()
    main(args)