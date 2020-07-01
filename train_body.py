import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import time
import argparse

from dataloader import load_dataloader
from model import Net
from misc import try_gpu, Arguments
from train_utils import set_seed, save_checkpoint, Loss

def main(args):    
    set_seed(args)
    save_dir = os.path.join(args.CHK_DIR, args.LOG_DIR, args.train_id)
    log_path = os.path.join('runs/', args.LOG_DIR, args.train_id)
    os.makedirs(save_dir,exist_ok=True)    
    os.makedirs(log_path,exist_ok=True)    
    ## save argparse parameters
    with open(os.path.join(log_path, args.train_id+'_args.yaml'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))        
    # writer = SummaryWriter(os.path.join('runs/',args.LOG_DIR, args.train_id))    
    writer = SummaryWriter(log_path)    

    train_loader = load_dataloaer(args)
    model = Net()
    model = try_gpu(model)
    model.train()
    criterion = Loss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=args.lr, weight_decay=args.weight_decay)               

    init_epoch = 0
    ## init time
    zero_time = time.time()

    for epoch in range(init_epoch, args.epochs):    
        start_time = time.time()

        # train_loss = train_per_epoch(train_loader, model, criterion, optimizer, epoch)
        avg_loss = 0
        for cnt, (img, target) in enumerate(train_loader, 1):
            print(cnt, img.shape, target.shape)
            img, target = try_gpu(img), try_gpu(target)
            pred = model(img)
            loss = criterion(pred, target)             

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item()

            if(cnt%10==0):
                # writer.add_scalar('training loss', loss.item(), epoch+cnt/(len(train_loader)//args.batch_size)) 
                writer.add_scalar('training loss', loss.item(), epoch*(len(train_loader)//args.batch_size)+cnt)                                 

            # if(cnt%100==0):
            if(cnt%5000==0):
                cp_file = os.path.join(save_dir, 'epoch_'+str(epoch)+'_itr_'+str(cnt)) + '.pt'
                save_checkpoint({ 'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), }, cp_file)

        avg_loss /= len(train_loader)
        end_time = time.time()

        epoch_time = end_time - start_time
        total_time = end_time - zero_time

        # writer.add_scalar('training loss', train_loss, epoch) 
        # total_train_loss.append(train_loss)
        total_train_loss.append(avg_loss)
        writer.close()
        
def train_per_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()

    avg_loss = 0
    for cnt, (img, heatmap) in enumerate(dataloader, 1):
        img, heatmap = try_gpu(img), try_gpu(heatmap)
        pred = model(img)
        loss = criterion(pred, heatmap) 

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        avg_loss += loss.item()

    return avg_loss/cnt
