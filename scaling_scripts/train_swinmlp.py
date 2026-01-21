import argparse
import time
import yaml
import torch
import os
import csv
import torch.distributed as dist
import torch.optim as optim
import utils 
from perun import monitor

@monitor()
def train_one_epoch(
    config_dict, 
    epoch, 
    model, 
    loader, 
    optimizer, 
    loss_fn, 
    device, 
    rank
):
    model.train()
    epoch_loss = 0.0

    # Set epoch for sampler if distributed, to shuffle data differently each epoch
    # This ensures that each process sees a different shuffled version of the dataset
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(loader):
        # Load data to device
        images = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True)

        # Swin Output ist [B, 2, H, W] (Logits)
        optimizer.zero_grad()
        predictions = model(images) 
        loss = loss_fn(predictions, masks)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Sum loss across all processes from all GPUs
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
        epoch_loss += loss.item()

        if rank == 0 and i % 10 == 0:
            print(f"Epoch [{epoch}][{i}/{len(loader)}] Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)

def val_one_epoch(
    config_dict, 
    epoch, 
    model, 
    loader, 
    optimizer, 
    loss_fn, 
    device, 
    rank
):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Load data to device
            images = batch[0].to(device, non_blocking=True)
            masks = batch[1].to(device, non_blocking=True)

            # Swin Output ist [B, 2, H, W] (Logits)
            predictions = model(images) 
            loss = loss_fn(predictions, masks)

            # Sum loss across all processes from all GPUs
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
            epoch_loss += loss.item()

            if rank == 0 and i % 10 == 0:
                print(f"[VAL] Epoch [{epoch}][{i}/{len(loader)}] Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)


@monitor()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Initialize distributed training
    device, localid, rank, world_size = utils.init_cuda()

    if rank == 0:
        print(f"Starting Training with Config: {args.config_file}")

    # Create DDP group
    ddp_group = dist.group.WORLD # all GPUs are in the same group
    model = utils.build_swin_model(config, device, localid, rank, ddp_group)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    loss_fn = utils.build_loss_fn(config) #Use TopKDiceLoss

    train_loader, val_loader = utils.build_swin_dataloaders(config, model_parallel=False, rank=rank)

    num_epochs = config["training"]["epochs"]
    num_samples = len(train_loader.dataset)
    num_samples_val = len(val_loader.dataset)
    batch_size = config["training"]["train_batch"]
    learn_rate = config["training"]["lr"]
    save_dir = config["training"]["save_dir"]

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file = f"log_{config['data']['image_height']}_{config['training']['train_batch']}.csv"
        log_path = os.path.join(save_dir, log_file)
        if not os.path.isfile(log_path):
            with open(log_path, mode='w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    'Epoch',
                    'Resolution',
                    'Learning_Rate',
                    'Batch_Size', 
                    'Num_Train_Samples',
                    'Num_Val_Samples',
                    'Train_Loss', 
                    'Val_Loss',
                    'Train_Dice',
                    'Val_Dice', 
                    'Duration_Sec',
                    'Train_TP',
                    'Train_FP',
                    'Train_TN',
                    'Train_FN',
                    'Train_Precision',
                    'Train_Recall',
                ])

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(
            config, epoch, model, train_loader, optimizer, loss_fn, device, rank
        )
        train_dice = utils.check_accuracy(train_loader, model, device, rank)

        # wait for all processes to finish epoch before validating
        dist.barrier() 
        val_loss = val_one_epoch(
            config, epoch, model, val_loader, optimizer, loss_fn, device, rank
        )
        val_dice = utils.check_accuracy(val_loader, model, device, rank)
        train_metrics = utils.compute_confusion_metrics(train_loader, model, device)
        val_metrics   = utils.compute_confusion_metrics(val_loader, model, device)

        dice_tensor = torch.tensor(val_dice).to(device)
        dist.broadcast(dice_tensor, src=0)
        scheduler.step(dice_tensor.item())

        end_time = time.time()
        duration = end_time - start_time
        
        # Learning Rate Step
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Dice Score: {train_dice:.4f} | LR: {current_lr:.1e} | Time: {(time.time()-start_time)/60:.1f} min")
            
            log_file = f"log_{config['data']['image_height']}_{config['training']['train_batch']}.csv"
            log_path = os.path.join(save_dir, log_file)
            with open(log_path, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    int(epoch),
                    int(config['data']['image_height']),
                    learn_rate,
                    batch_size,
                    num_samples,
                    num_samples_val,
                    round(float(train_loss), 4), 
                    round(float(val_loss), 4),
                    round(float(train_dice), 4),
                    round(float(val_dice), 4), 
                    round(duration, 2),
                    int(train_metrics['TP']),
                    int(train_metrics['FP']),
                    int(train_metrics['TN']),
                    int(train_metrics['FN']),
                    round(float(train_metrics['Precision']), 4),
                    round(float(train_metrics['Recall']), 4),
                ])

           # utils.save_predictions_as_imgs(
                #val_loader, model,f"{config['training']['save_dir']}/images", 
                #device=device, epoch=epoch, rank=rank)

            # utils.save_model(model, optimizer,scheduler, config["training"]["save_dir"], rank, epoch)
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
