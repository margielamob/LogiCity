import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tasks.nav.neural import NaviNet
from tasks.nav.data import NaviDatasetPOV
from tasks.nav.pkl_parser import parse_pkl, visualize_traj, get_fov, move, SCALE
from sklearn.metrics import accuracy_score
from torchvision import transforms

# Define a transform to convert images to PyTorch tensors and normalize them
# Assuming the images are in the range [0, 255], you would normalize them to be in the range [0, 1]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy arrays to PILImage for further processing
    transforms.ToTensor(),    # Convert PILImage to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

def test(model, dataloader, args, logger, epoch, device, global_img, visualize=True):
    model.eval()
    controlled_acc = []
    navi_acc = []
    with torch.no_grad():
        for i, (fov_imgs, goals, labels, init_center, final_goal) in tqdm(enumerate(dataloader)):
            outputs = []
            # controlled classification score
            for j in range(len(fov_imgs)):
                fov_img = fov_imgs[j].to(device)
                goal = goals[j].to(device)
                output = model(fov_img, goal)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            labels = torch.cat(labels, dim=0).cpu().numpy()
            max_indices = torch.argmax(outputs, dim=1)
            predictions = F.one_hot(max_indices, num_classes=outputs.shape[1]).cpu().numpy()
            accuracy = accuracy_score(labels, predictions)
            controlled_acc.append(accuracy)
            # navi classification score, just use final goal
            if not visualize:
                navi_acc.append(0)
            else:
                num_steps = len(labels)
                step = 0
                reach_goal = False
                img = fov_imgs[0].to(device)
                goal = goals[0].to(device)
                # visualize the trajectory on global map
                init_center = init_center[0].cpu().numpy().astype(np.int64)
                curr_center = init_center.copy()
                final_goal = final_goal[0].cpu().numpy().astype(np.int64)
                cpu_goal = goal[0].cpu().numpy()
                cpu_goal = np.array([cpu_goal[0]*(args.fov//2), cpu_goal[1]*(args.fov//2)]).astype(np.int64)
                cpu_goal = cpu_goal + init_center
                vis_global = global_img.copy()
                vis_global[init_center[0]:init_center[0]+SCALE, init_center[1]:init_center[1]+SCALE] = np.array([0, 110, 255])
                vis_global[final_goal[0]:final_goal[0]+SCALE, final_goal[1]:final_goal[1]+SCALE] = np.array([255, 0, 0])
                save_vis = vis_global.copy()
                save_vis[cpu_goal[0]:cpu_goal[0]+SCALE, cpu_goal[1]:cpu_goal[1]+SCALE] = np.array([255, 175, 0])
                vis_fov = get_fov(save_vis, init_center, args.fov)
                os.makedirs(f'{args.log_dir}/{args.exp}/nav_vis/{epoch}/{i}', exist_ok=True)
                cv2.imwrite(f'{args.log_dir}/{args.exp}/nav_vis/{epoch}/{i}/init.png', save_vis)
                cv2.imwrite(f'{args.log_dir}/{args.exp}/nav_vis/{epoch}/{i}/init_fov.png', vis_fov)
                # Create a writer object for the video
                width_of_combined_image = vis_global.shape[1] + vis_fov.shape[1] + 20
                height_of_combined_image = max(vis_global.shape[0], vis_fov.shape[0])
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Video codec
                video_path = f'{args.log_dir}/{args.exp}/nav_vis/{epoch}/{i}/trajectory.mp4'
                video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (width_of_combined_image, height_of_combined_image))
                while step < 2*num_steps:
                    output = model(img, goal)
                    max_index = torch.argmax(output, dim=1).item()
                    curr_center = move(curr_center, max_index)
                    if np.all(curr_center == final_goal):
                        reach_goal = True
                        break
                    # visualize the trajectory on global map
                    vis_global[curr_center[0]:curr_center[0]+SCALE, curr_center[1]:curr_center[1]+SCALE] = np.array([0, 110, 255])
                    local_goal = np.array([final_goal[0]-curr_center[0], final_goal[1]-curr_center[1]])
                    local_goal = np.clip(local_goal, -args.fov//2+2, args.fov//2-2)
                    cpu_goal = local_goal+curr_center
                    save_vis = vis_global.copy()
                    # visualize the local goal
                    save_vis[cpu_goal[0]:cpu_goal[0]+SCALE, cpu_goal[1]:cpu_goal[1]+SCALE] = np.array([255, 175, 0])
                    local_goal = local_goal / (args.fov//2)
                    vis_fov = get_fov(save_vis, curr_center, args.fov)
                    combined_img = np.zeros((height_of_combined_image, width_of_combined_image, 3), dtype=np.uint8)
                    combined_img[:vis_global.shape[0], :vis_global.shape[1]] = save_vis
                    combined_img[:vis_fov.shape[0], vis_global.shape[1]+10:vis_global.shape[1]+10+vis_fov.shape[1]] = vis_fov
                    cv2.putText(combined_img, f'Step: {step}', (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
                    video_writer.write(combined_img)
                    goal = torch.from_numpy(local_goal).to(torch.float32).unsqueeze(0).to(device)
                    img = IMAGE_TRANSFORM(vis_fov).unsqueeze(0).to(device)
                    step += 1
                if reach_goal:
                    logger.info(f'Test Seq {i}, Success! Goal reached in {step+1}/{num_steps} steps')
                    navi_acc.append(num_steps/(step+1))
                else:
                    logger.info(f'Test Seq {i}, Failed.')
                    navi_acc.append(0)
                video_writer.release()
    return np.mean(controlled_acc), np.mean(navi_acc)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

def runner(args, logger, writer):
    # Set up logging
    os.makedirs(f'{args.log_dir}/{args.exp}', exist_ok=True)

    # parse raw pkl for all data
    train_data, test_data, global_img = parse_pkl(args.raw_pkl_path, logger, args.fov, training_ratio=0.9)
    action_space = train_data["labels"][0]["action"].shape[0]
    # check_traj = np.random.choice(train_data['traj'], 1)[0]
    # visualize_traj(global_img, train_data, check_traj)
    train_data = NaviDatasetPOV(global_img, train_data, logger, args.fov, True, IMAGE_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # Load test data
    test_data = NaviDatasetPOV(global_img, test_data, logger, args.fov, False, IMAGE_TRANSFORM)
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
            controlled_accuracy, succ_rate = test(model, test_loader, args, logger, epoch+1, args.device, global_img, visualize=True)
            # Log individual label losses
            writer.add_scalar(f'Accuracy/controlled', controlled_accuracy, epoch)
            writer.add_scalar(f'Accuracy/navi_succ', succ_rate, epoch)
            logger.info(f'Epoch: {epoch}, Test Acc: {controlled_accuracy}')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=f'{args.log_dir}/{args.exp}/checkpoint_{epoch+1}.pth.tar')