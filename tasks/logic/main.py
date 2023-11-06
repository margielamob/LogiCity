import torch
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from data import LogicDataset
from models import MLP
from utils import parse_pkl
import matplotlib.pyplot as plt

def visualize_samples(data_X, data_Y, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2*num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(data_X[i].reshape(1, -1), cmap='gray')
        axes[i, 0].set_title('Input Data Sample')
        axes[i, 1].imshow(data_Y[i].reshape(-1, 2), cmap='gray')
        axes[i, 1].set_title('Output Data Sample')
    plt.tight_layout()
    plt.show()

def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    # parse raw pkl for training
    input_sz, output_sz, data_X, data_Y = parse_pkl(args.data_path)

    # Load or create model
    model = MLP(input_sz, output_sz) if not args.model_path else torch.load(args.model_path)

    # Set up logging
    writer = SummaryWriter(args.log_dir)

    # Visualize some data samples
    visualize_samples(data_X[:5], data_Y[:5])

    # Load training data
    train_data = LogicDataset(data_X, data_Y)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
    
    # Save the trained model
    torch.save(model, 'trained_mlp.pth')

    # Load test data
    test_input_sz, test_output_sz, test_data_X, test_data_Y = parse_pkl(args.test_data_path)
    test_data = LogicDataset(test_data_X, test_data_Y)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Testing loop
    test_loss = test(model, test_loader, criterion)
    writer.add_scalar('Loss/test', test_loss)
    print(f'Test Loss: {test_loss}')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to load the model')
    parser.add_argument('--data_path', type=str, default='log/easy_1k_train.pkl', help='Path to the training data')
    parser.add_argument('--test_data_path', type=str, default='log/easy_1k_test.pkl', help='Path to the test data')
    parser.add_argument('--log_dir', type=str, default='logs/logic', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    args = parser.parse_args()
    main(args)
