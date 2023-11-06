import torch
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from tasks.logic.data import LogicDataset
from tasks.logic.models import MLP
from tasks.logic.pkl_parser import parse_pkl
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_samples(data_X, data_Y, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(data_X[i].reshape(1, -1), cmap='gray')
        axes[i, 0].set_title('Input Data Sample')
        axes[i, 1].imshow(data_Y[i].reshape(-1, 2), cmap='gray')
        axes[i, 1].set_title('Output Data Sample')
    plt.tight_layout()
    plt.show()

def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

def main(args):
    # Set up logging
    writer = SummaryWriter(args.log_dir)

    # parse raw pkl for training
    input_sz, output_sz, data_X, data_Y = parse_pkl(args.data_path)

    # Load or create model
    if args.resume:
        checkpoint = load_checkpoint(args.model_path)
        model = MLP(input_sz, output_sz)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        model = MLP(input_sz, 128, output_sz).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        start_epoch = 0

    # Load training data
    train_data = LogicDataset(data_X, data_Y)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # Load test data
    _, _, test_data_X, test_data_Y = parse_pkl(args.test_data_path)
    test_data = LogicDataset(test_data_X, test_data_Y)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Define loss function
    criterion = nn.MSELoss().to(args.device)

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
        test_loss = test(model, test_loader, criterion, args.device)
        writer.add_scalar('Loss/test', test_loss)
        print(f'Epoch: {epoch}, Test Loss: {test_loss}')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=f'checkpoint_{epoch}.pth.tar')

    # [Test loop from previous example]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on')
    parser.add_argument('--data_path', type=str, default='log/easy_1k_train.pkl', help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default='log/easy_100_test.pkl', help='Path to the test data')
    parser.add_argument('--log_dir', type=str, default='log/logic', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    args = parser.parse_args()
    main(args)
