import os
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from tasks.nav.neural import NaviNet
from tasks.nav.data import NaviDatasetPOV
from tasks.nav.pkl_parser import parse_pkl, visualize_traj
from torchvision import transforms

# Define a transform to convert images to PyTorch tensors and normalize them
# Assuming the images are in the range [0, 255], you would normalize them to be in the range [0, 1]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy arrays to PILImage for further processing
    transforms.ToTensor(),    # Convert PILImage to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

def test(model, dataloader, args, epoch, device, Yname, save=False):
    model.eval()
    # TODO: add test code
    return

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

def runner(args, logger, writer):
    # Set up logging
    os.makedirs(f'{args.log_dir}/{args.exp}', exist_ok=True)

    # parse raw pkl for all data
    train_data, test_data, global_img = parse_pkl(args.raw_pkl_path, logger, args.fov, training_ratio=0.8)
    action_space = train_data["labels"][0]["action"].shape[0]
    # check_traj = np.random.choice(train_data['traj'], 1)[0]
    # visualize_traj(global_img, train_data, check_traj)
    train_data = NaviDatasetPOV(global_img, train_data, logger, 224, True, IMAGE_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # Load test data
    test_data = NaviDatasetPOV(global_img, test_data, logger, 224, False, IMAGE_TRANSFORM)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Load or create model
    if args.resume:
        logger.info(f'Resuming from {args.model_path}')
        checkpoint = load_checkpoint(args.model_path)
        model = NaviNet(args.backbone, 
                        args.pos_hidden, 
                        args.decoder_hidden, 
                        action_space).to(args.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)
        for param in model.backbone.parameters():
            param.requires_grad = False  # Freeze the backbone parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        logger.info('Creating new model, training from scratch')
        model = NaviNet(args.backbone, 
                        args.pos_hidden, 
                        args.decoder_hidden, 
                        action_space).to(args.device)
        if args.backbone_lr == 0:
            # Freeze the backbone parameters
            for param in model.backbone.parameters():
                param.requires_grad = False
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        else:
            backbone_lr = args.learning_rate * args.backbone_lr
            backbone_params = list(model.backbone.parameters())
            other_params = [param for param in model.parameters() if param not in backbone_params]
            optimizer = optim.Adam(
                        [{'params': backbone_params, 'lr': backbone_lr},
                        {'params': other_params, 'lr': args.learning_rate}],
                        weight_decay=args.weight_decay
                        )
        start_epoch = 0
    
    # Define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Training loop
    model.train()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        for i, (imgs, goals, labels) in enumerate(train_loader):
            imgs = imgs.to(args.device)
            goals = goals.to(args.device)
            labels = labels.to(args.device)
            labels = torch.argmax(labels, dim=1)
            optimizer.zero_grad()
            outputs = model(imgs, goals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
        if (epoch + 1) % args.save_freq == 0:
            controlled_accuracy, succ_rate = test(model, test_loader, args, epoch+1, args.device, save=epoch==args.epochs-1)
            # Log individual label losses
            writer.add_scalar(f'Accuracy/controlled', controlled_accuracy, epoch)
            writer.add_scalar(f'Accuracy/navi_succ', succ_rate, epoch)
            logger.info(f'Epoch: {epoch}, Test Acc: {controlled_accuracy}')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.log_dir}/{args.exp}/checkpoint_{epoch}.pth.tar')