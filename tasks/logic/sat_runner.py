import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

import satnet
from tasks.logic.data import LogicDatasetSAT
from tasks.logic.pkl_parser import parse_pkl

def test(model, dataloader, args, epoch, device, Yname, save=False):
    model.eval()
    with torch.no_grad():
        outputs = []
        Ys = []
        for i, (inputs, masks, labels) in tqdm(enumerate(dataloader)):
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            output = model(inputs, masks)
            outputs.append(output[:, -labels.shape[1]:])
            Ys.append(labels)
        outputs = torch.cat(outputs, dim=0)
        Ys = torch.cat(Ys, dim=0)
    # Average the loss for each label
    data_Y_converted = Ys.cpu().numpy()
    # Get the indices of the max probabilities
    max_indices = torch.argmax(outputs, dim=1)

    # Convert to one-hot encoded tensor
    predictions = F.one_hot(max_indices, num_classes=outputs.shape[1]).cpu().numpy()
    accuracy = accuracy_score(data_Y_converted, predictions)
    report_dict = classification_report(data_Y_converted, predictions, target_names=Yname, output_dict=True)
    if save:
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_excel(f"{args.log_dir}/{args.exp}/test_results_epoch_{epoch}.xlsx", index=True)

    return accuracy, report_dict

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

def runner(args, logger, writer):
    # Set up logging
    os.makedirs(f'{args.log_dir}/{args.exp}', exist_ok=True)

    # Load training data
    # parse raw pkl for training
    input_sz, output_sz, data_X, data_Y, Xname, Yname = parse_pkl(args.train_data_path, logger)
    train_data = LogicDatasetSAT(data_X, data_Y, Xname, Yname, logger, False, args.uni_boundary, args.w_bernoulli)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # Load test data
    _, _, test_data_X, test_data_Y, _, _ = parse_pkl(args.test_data_path, logger)
    test_data = LogicDatasetSAT(test_data_X, test_data_Y, Xname, Yname, logger, args.w_bernoulli)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load or create model
    if args.resume:
        logger.info(f'Resuming from {args.model_path}')
        checkpoint = load_checkpoint(args.model_path)
        model = satnet.SATNet(input_sz+output_sz, args.m, args.aux)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger.info('Creating new model, training from scratch')
        model = satnet.SATNet(input_sz+output_sz, args.m, args.aux).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        start_epoch = 0

    # Training loop
    model.train()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        for i, (inputs, masks, labels) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = F.binary_cross_entropy(outputs[:, -output_sz:], labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
        if (epoch + 1) % args.save_freq == 0:
            logger.info(f'Testing and Saving model at epoch {epoch}')
            test_accuracy, test_label_acc = test(model, test_loader, args, epoch+1, args.device, Yname, save=True)
            # Log individual label losses
            writer.add_scalar(f'Loss/test_acc', test_accuracy, epoch)
            for label, acc in test_label_acc.items():
                writer.add_scalar(f'Error/test_{label}', acc['precision'], epoch)
            logger.info(f'Epoch: {epoch}, Test Acc: {test_accuracy}')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.log_dir}/{args.exp}/checkpoint_{epoch}.pth.tar')