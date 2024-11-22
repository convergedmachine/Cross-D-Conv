import os
import time
import utils
import torch
import datetime
from get_args_parser import get_args_parser
from load_data import load_data
from create_model import create_model
from get_optim import get_optim
from train_one_epoch import train_one_epoch
from evaluate import evaluate
from torchinfo import summary
from collections import OrderedDict

torch.set_float32_matmul_precision('high')

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
        
    data_loader, data_loader_test, train_sampler, test_sampler, num_classes = load_data(args)
    device = torch.device(args.device)

    model = create_model(args).to(device)
    print(model)
    
    criterion, optimizer, lr_scheduler = get_optim(model, args)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        args = checkpoint["args"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
    elif args.finetune:
        checkpoint = torch.load(args.finetune, map_location=device, weights_only=False)["model"]
        model.load_state_dict(checkpoint)    
    
    print(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        model_without_ddp = model.module
            
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, args, device, epoch)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, args, device)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if epoch > args.epochs - 4:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))                   
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)