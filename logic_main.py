import os
import yaml
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from tasks.logic.models import MLP
from tasks.logic.data import LogicDataset
from tasks.logic.pkl_parser import parse_pkl

def update_args_from_config(args, config):
    for key, value in config.items():
        setattr(args, key, value)

def setup_logging(args):
    log_file_path = f"{args.log_dir}/{args.exp}/config.log"
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logging.info("Configuration Parameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

def visualize_samples(data_X, data_Y, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(data_X[i].reshape(1, -1), cmap='gray')
        axes[i, 0].set_title('Input Data Sample')
        axes[i, 1].imshow(data_Y[i].reshape(-1, 2), cmap='gray')
        axes[i, 1].set_title('Output Data Sample')
    plt.tight_layout()
    plt.show()

def test(model, dataloader, args, epoch, device, save=False):
    model.eval()
    label_losses = {name: 0.0 for name in dataloader.dataset.Yname}  # Initialize losses for each Yname
    label_fail = {name: 0.0 for name in dataloader.dataset.Yname}  # Initialize losses for each Yname
    
    # To store the number of times each label was encountered
    label_counts = {name: 0 for name in dataloader.dataset.Yname}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            outputs_label = outputs.argmax(dim=1)
            labels_pos = labels.argmax(dim=1)
            # Calculate loss for each label
            for i, name in enumerate(dataloader.dataset.Yname):
                ids = (labels[:, i] == 1.0).nonzero()
                if ids.shape[0]>0:
                # Note: this assumes that the model's output and the labels are structured
                # such that each column corresponds to the label represented by Yname[i]
                    sum_label_loss = torch.abs(outputs[ids, i] - labels.to(device)[ids, i]).sum().item()
                    sum_fail = (outputs_label[ids] != labels_pos.to(device)[ids]).sum().item()
                    label_losses[name] += sum_label_loss
                    label_counts[name] += ids.shape[0]
                    label_fail[name] += sum_fail
    
    # Average the loss for each label
    for name in label_losses:
        if label_counts[name] == 0:
            label_losses[name] = -1.0
            label_fail[name] = -1.0
        else:
            label_losses[name] /= label_counts[name]
            label_fail[name] /= label_counts[name]
    if save:
        df = pd.DataFrame({
            'Label': list(label_losses.keys()),
            'Loss': list(label_losses.values()),
            'Error': list(label_fail.values())
        })
        df.to_excel(f"{args.log_dir}/{args.exp}/test_results_epoch_{epoch}.xlsx", index=False)

    return label_losses, label_fail

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

def main(args):
    # Set up logging
    os.makedirs(f'{args.log_dir}/{args.exp}', exist_ok=True)
    setup_logging(args)
    writer = SummaryWriter(f'{args.log_dir}/{args.exp}')

    # parse raw pkl for training
    input_sz, output_sz, data_X, data_Y, Xname, Yname = parse_pkl(args.data_path)

    # Load or create model
    if args.resume:
        checkpoint = load_checkpoint(args.model_path)
        model = MLP(input_sz, 128, output_sz)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        model = MLP(input_sz, 128, output_sz).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        start_epoch = 0

    # Load training data
    train_data = LogicDataset(data_X, data_Y, Xname, Yname, args.adjust_dist)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # Load test data
    _, _, test_data_X, test_data_Y, _, _ = parse_pkl(args.test_data_path)
    test_data = LogicDataset(test_data_X, test_data_Y, Xname, Yname)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Define loss function
    criterion = nn.L1Loss().to(args.device)

    # Training loop
    model.train()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

        test_label_losses, test_label_error = test(model, test_loader, args, epoch+1, args.device, save=(epoch + 1) % args.save_freq == 0)
        # Log individual label losses
        avg_loss = 0
        num_labels = 0
        avg_error = 0
        for label, loss in test_label_losses.items():
            if loss>=0:
                avg_loss += loss
                num_labels += 1
                avg_error += test_label_error[label]
            writer.add_scalar(f'Loss/test_{label}', loss, epoch)
            writer.add_scalar(f'Error/test_{label}', test_label_error[label], epoch)
        print(f'Epoch: {epoch}, Test Loss: {avg_loss/num_labels}, Test Error: {avg_error/num_labels}')
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.log_dir}/{args.exp}/checkpoint_{epoch}.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/tasks/logic/easy_small_unbalanced.yaml', help='Directory to configure file')
    parser.add_argument('--resume', default=False, help='Resume training from a checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for the dataloader')
    parser.add_argument('--data_path', type=str, default='log/easy_1k_train.pkl', help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default='log/easy_100_test.pkl', help='Path to the test data')

    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Update args with config
    update_args_from_config(args, config)
    main(args)
