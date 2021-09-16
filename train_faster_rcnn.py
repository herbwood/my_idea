import os
import gc
import time
import wandb
import logging 

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms 

from dataset import CrowdHumanDataset
from faster_rcnn import fasterrcnn_resnet50_fpn
from task_utils import TqdmLoggingHandler, write_log, optimizer_select, scheduler_select


def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, logger, device):
    
    start_time_e = time.time()
    model = model.train()
    train_loss = 0

    for i, (data, target) in enumerate(dataloader):

        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)

        losses, detections = model(data)
        total_losses = sum([losses[key].means() for key in losses.keys()])
        assert torch.isinfinite(total_losses).all(), losses

        total_losses.backward()
        optimizer.step()

        if args.scheduler in ['constant', 'warmup']:
            scheduler.step()
        if args.scheduler == 'reduce_train':
            scheduler.step(total_losses)

        if i == 0 or freq == args.print_freq or i == len(dataloader):
            batch_log = ""
            write_log(logger, batch_log)
            freq = 0
        freq += 1
        
        train_loss += total_losses.item()
    
    return train_loss

def valid_epoch(args, model, dataloader, device):

    model = model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            
            data = data.to(device)
            target = target.to(device)

            losses, detections = model(data)
            total_losses = sum([losses[key].means() for key in losses.keys()])
            assert torch.isinfinite(total_losses).all(), losses

            val_loss += total_losses.item()

    return val_loss

def faster_rcnn_training(args):

    # wandb.init()
    # wandb.run.name = "Faster_R-CNN_Training"
    # wandb.config.update(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(' %(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.propagate = False
    
    #===================================#
    #============Data Load==============#
    #===================================#

    write_log(logger, 'Load data...')
    gc.disable()
    # transform_dict = {
    #     'train' : transforms.Compose(
    #         [
    #             transforms.Resize((args.img_height, args.img_width)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]),
    #     'valid' : transforms.Compose(
    #         [
    #             transforms.Resize((args.height, args.width)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ])
    # }

    dataset_dict = {
        'train' : CrowdHumanDataset(root_dir=args.data_path, phase='train', transform=None),
        'val' : CrowdHumanDataset(root_dir=args.data_path, phase='val', transform=None)
    }

    dataloader_dict = {
        'train' : DataLoader(dataset_dict['train'], drop_last=True, batch_size=args.batch_size, 
                             shuffle=True, pin_memory=True, num_workers=args.num_workers, collate_fn=dataset_dict['train'].merge_batch),
        'val' : DataLoader(dataset_dict['val'], drop_last=False, batch_size=args.batch_size, 
                             shuffle=False, pin_memory=True, num_workers=args.num_workers)
    }

    gc.enable()
    write_log(logger, f"Total number of training sets and iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    write_log(logger, "Instantiating models...")
    model = fasterrcnn_resnet50_fpn(pretrained=False, progress=True, 
                                    num_classes=91, pretrained_backbone=True, 
                                    trainable_backbone_layers=None)
    model.to(device)

    optimizer = optimizer_select(model, args)
    scheduler = scheduler_select(optimizer, dataloader_dict, args)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.train()
        model = model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    # best_val_acc = 0
    best_train_loss = 1e+5
    
    # wandb.watch(model)
    write_log(logger, 'Train start!')
    for epoch in range(start_epoch, args.num_epochs):

        train_loss = train_epoch(args, epoch, model, dataloader_dict['train'], optimizer, scheduler, logger, device)
        train_loss /= len(dataloader_dict['valid'])
        # val_loss = valid_epoch(args, model, dataloader_dict['valid'], device)
        # val_loss /= len(dataloader_dict['valid'])

        write_log(logger, f"Train Loss : {train_loss:3.3f}")
        # wandb.log({"Train Loss" : train_loss})

        if train_loss < best_train_loss:
            write_log(logger, "Checkpoint saving...")
            torch.save({
                'epoch' : epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, os.path.join(args.save_path, 'checkpoint.pth.tar'))
            best_train_loss = train_loss
            best_epoch = epoch
        
        else:
            # else_log = f"Still {best_epoch} epoch Accuracy ({round(best_val_acc, 2)})% is better..."
            else_log = f"Still {best_epoch} epoch Accuracy ({round(best_train_loss, 2)})% is better..."
            write_log(logger, else_log)

        print(f"Best Epoch : {best_epoch+1}")
        # print(f"Best Accuracy : {round(best_val_acc, 2)}")
        print(f"Best Loss : {round(best_train_loss, 3)}")

    

    