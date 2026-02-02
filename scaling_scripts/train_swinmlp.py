import argparse
import time
import yaml
import torch
import os
import csv
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler
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
    sheduler,
    loss_fn, 
    device, 
    rank,
    scaler,
    prof = None
):
    model.train()
    running_loss = 0.0
    total_samples = 0
    

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

        with torch.amp.autocast("cuda", dtype=torch.float16): 
            predictions = model(images) 
            loss = loss_fn(predictions, masks)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Sum loss across all processes from all GPUs
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
        current_batch_size = images.size(0)
        running_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        if prof:
            prof.step()

        if rank == 0 and i % 10 == 0:
            print(f"Epoch [{epoch}][{i}/{len(loader)}] Loss: {loss.item():.4f}")

    return running_loss / total_samples

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
    running_loss = 0.0
    total_samples = 0

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
            current_batch_size = images.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

            if rank == 0 and i % 10 == 0:
                print(f"[VAL] Epoch [{epoch}][{i}/{len(loader)}] Loss: {loss.item():.4f}")

    return running_loss / total_samples


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
    train_loader, val_loader = utils.build_swin_dataloaders(config, model_parallel=False, rank=rank)
    num_epochs = config["training"]["epochs"]
    num_samples = len(train_loader.dataset)
    num_samples_val = len(val_loader.dataset)
    batch_size = config["training"]["train_batch"]
    learn_rate = config["training"]["lr"]
    save_dir = config["training"]["save_dir"]
    scaler = torch.amp.GradScaler("cuda")

    optimizer = optim.AdamW(
        model.parameters(),
        lr= learn_rate,
        weight_decay=1e-4,
        betas = (0.9, 0.999)
    )
    
    
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3,    
        factor=0.5,   
        threshold=1e-3,
        min_lr=1e-6, 
    )

    """ scheduler = OneCycleLR(
        optimizer,
        max_lr=learn_rate, 
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,  
        final_div_factor=1000.0
    ) """
    loss_fn = utils.build_loss_fn(config) #Use TopKDiceLoss

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file = f"log_{config['data']['image_height']}_{num_epochs}_{learn_rate}.csv"
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
                    'Val_Loss_All',
                    'Train_Dice',
                    'Val_Dice', 
                    'Duration_Sec',
                    'Train_TP',
                    'Train_FP',
                    'Train_TN',
                    'Train_FN',
                    'Train_Precision',
                    'Train_Recall',
                    'Val_TP',
                    'Val_FP',
                    'Val_TN',
                    'Val_FN',
                    'Val_Precision',
                    'Val_Recall'
                ])

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(save_dir),
        record_shapes=True,
        with_stack=True
    ) as prof:
        
        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = train_one_epoch(
                config, epoch, model, train_loader, optimizer, scheduler, loss_fn, device, rank, scaler=scaler, prof=prof
            )
            # wait for all processes to finish epoch before validating
            dist.barrier() 
            train_metrics = utils.evaluate(train_loader, model,loss_fn, device, rank)
            val_metrics   = utils.evaluate(val_loader, model, loss_fn, device, rank)

            if rank == 0:
                current_val_dice = val_metrics["Dice"]
            else:
                current_val_dice = 0.0

            dice_tensor = torch.tensor(current_val_dice).to(device)
            dist.broadcast(dice_tensor, src=0)
            scheduler.step(dice_tensor.item())

            end_time = time.time()
            duration = end_time - start_time
            
            # Learning Rate Step
            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                train_dice = train_metrics["Dice"]
                val_loss = val_metrics["Loss"]
                val_dice = val_metrics["Dice"]
                val_loss_all = val_metrics["Loss_All"]

                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Dice Score: {train_dice:.4f} | LR: {current_lr:.1e} | Time: {(time.time()-start_time)/60:.1f} min")
                
                #log_file = f"log_{config['data']['image_height']}_{num_epochs}_{learn_rate}.csv"
                #log_path = os.path.join(save_dir, log_file)
                with open(log_path, mode='a', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow([
                        int(epoch),
                        int(config['data']['image_height']),
                        current_lr,
                        batch_size,
                        num_samples,
                        num_samples_val,
                        round(float(train_loss), 4), 
                        round(float(val_loss), 4),
                        round(float(val_loss_all), 4),
                        round(float(train_dice), 4),
                        round(float(val_dice), 4), 
                        round(duration, 2),
                        round(float(train_metrics['TP']), 4),
                        round(float(train_metrics['FP']), 4),
                        round(float(train_metrics['TN']), 4),
                        round(float(train_metrics['FN']), 4),
                        round(float(train_metrics['Precision']), 4),
                        round(float(train_metrics['Recall']), 4),
                        round(float(val_metrics['TP']), 4),
                        round(float(val_metrics['FP']), 4),
                        round(float(val_metrics['TN']), 4),
                        round(float(val_metrics['FN']), 4),
                        round(float(val_metrics['Precision']), 4),
                        round(float(val_metrics['Recall']), 4)
                    ])

            # utils.save_predictions_as_imgs(
                    #val_loader, model,f"{config['training']['save_dir']}/images", 
                    #device=device, epoch=epoch, rank=rank)

                # utils.save_model(model, optimizer,scheduler, config["training"]["save_dir"], rank, epoch)
            dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
