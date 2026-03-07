import argparse
import time
import yaml
import os
import csv

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.profiler import ProfilerActivity
from perun import monitor

from Jigsaw import utils as jigsaw_utils
from mlpmixer_ctimages.loss import CETopKDiceLoss
import utils

torch.set_float32_matmul_precision('high')

def setup_groups(rank, n_parallel):
    # Set up model  groups
    match n_parallel:
        case 1:
            model_group, ddp_group, spatial_group, channel_group = (
                jigsaw_utils.setup_groups(rank, n_parallel, 1, 1)
            )
        case 2:
            model_group, ddp_group, spatial_group, channel_group = (
                jigsaw_utils.setup_groups(rank, n_parallel, 1, 2)
            )
        case 4:
            model_group, ddp_group, spatial_group, channel_group = (
                jigsaw_utils.setup_groups(rank, n_parallel, 2, 2)
            )

    return model_group, ddp_group, spatial_group, channel_group

@monitor()
def train_one_epoch(
    model, 
    loader, 
    optimizer, 
    loss_fn, 
    device, 
    profiler = None,
):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True)

        #Forward Pass
        if masks.dim() == 3:
            masks = masks.unsqueeze(1).float()
        else:
            masks = masks.float()

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16): 
            predictions = model(images) 
            loss = loss_fn(predictions.float(), masks)

        # Backward Pass
        loss.backward()

        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        current_batch_size = images.size(0)
        running_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        if profiler:
            profiler.step()

    return running_loss / total_samples


@monitor()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Initialize distributed training
    device, localid, rank, world_size = jigsaw_utils.init_cuda()

    # Set up model groups for parallelism
    n_parallel = config["model_parallel"]["parallelism"]
    model_group, ddp_group, spatial_group, channel_group = setup_groups(rank, n_parallel)

    num_epochs     = config["training"]["epochs"]
    batch_size     = config["training"]["train_batch"]
    learn_rate     = config["training"]["lr"]
    save_dir       = config["training"]["save_dir"]
    kernel_flex    = config["model"]["kernel_flex"]
    save_model     = config["model"]["save_model"]
    checkpoint_path = config["model"]["checkpoint_file"]
    seed           = config["model"]["seed"]
    image_height   = config["data"]["image_height"]
    image_width    = config["data"]["image_width"]

    kernel_str = "flex" if kernel_flex else "fixed"

    # Set all seeds in the same model group to be different
    jigsaw_utils.set_all_seeds(1 + rank % n_parallel)

    # Set random seed and ensure the same model groups same seeds
    # Set all GPUs to the same seed so that they read the same data
    jigsaw_utils.set_all_seeds_model_group(1, rank=rank, n_parallel=n_parallel)
    
    train_loader, val_loader, train_sampler = utils.build_dataloaders(
        config_dict=config,
        rank=rank,
        n_parallel=n_parallel,
        world_size=world_size
    )

    model = utils.build_model(config_dict=config, device=device,
                            localid=localid, rank=rank,
                            n_parallel=n_parallel, model_group=model_group,
                            ddp_group=ddp_group,spatial_group=spatial_group,
                            channel_group=channel_group)

    optimizer = optim.AdamW(
        model.parameters(),
        lr= learn_rate,
        weight_decay=1e-4,
        betas = (0.9, 0.999),
        eps=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=4,    
        factor=0.8,   
        threshold=1e-3,
        min_lr=1e-6, 
    )

    loss_fn = utils.build_loss_fn(config)
    loss_fn_all = CETopKDiceLoss(k=100, smooth=1e-5)

    num_samples = len(train_loader.dataset)
    num_samples_val = len(val_loader.dataset)
    
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log_file = f"log_{image_height}_{num_epochs}_{learn_rate}_{kernel_str}_{seed}_jigsaw.csv"
        log_path = os.path.join(save_dir, log_file)

        if not os.path.isfile(log_path):
            with open(log_path, mode='w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    'Num_Params',
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
                    'Peak_VRAM_GB', 
                    'FLOPs_per_image',
                    'Latency_ms_per_image',
                    'Throughput_img_per_sec',
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

    trace_dir = os.path.join(
    save_dir, 
    f"trace_{image_height}_{num_epochs}_{learn_rate}_{kernel_str}_{seed}")

    if rank == 0:
        os.makedirs(trace_dir, exist_ok=True)

    # Calculate the number of batches per epoch to determine the profiling schedule
    batches_per_epoch = len(train_loader)
    wait_steps = max(0, batches_per_epoch - 6)
    best_val_dice = 0.0

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait_steps, warmup=3, active=3, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        with_flops=True,
        record_shapes=True,
        with_modules=True,
        profile_memory=True,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:  
        
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler to shuffle data differently each epoch
            train_sampler.set_epoch(epoch)

            torch.cuda.reset_peak_memory_stats() 
            torch.cuda.synchronize()
            start_time = time.time()

            train_loss = train_one_epoch(
                model= model, loader=train_loader, optimizer=optimizer,
                loss_fn=loss_fn, device=device, profiler=prof)
            
            if rank % n_parallel == 0:
                train_metrics = utils.evaluate(train_loader, model, loss_fn, loss_fn_all, device)
                val_metrics   = utils.evaluate(val_loader,   model, loss_fn, loss_fn_all, device)
        
                dice_tensor = torch.tensor(val_metrics["Dice"]).to(device)
                dist.all_reduce(dice_tensor, op=dist.ReduceOp.AVG, group=ddp_group)
                val_metrics["Dice"] = dice_tensor.item()

                scheduler.step(val_metrics["Dice"])
            
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time

            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
            latency_ms = (duration / num_samples) * 1000
            throughput = num_samples / duration

            if epoch <= 2:
                raw_flops = sum(item.flops for item in prof.key_averages())
                total_flops = raw_flops / (3 * batch_size)
            else:
                total_flops = 0

            if rank == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Dice: {val_metrics['Dice']:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Duration: {duration:.1f}s"
                )

                with open(log_path, mode='a', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow([
                        sum(p.numel() for p in model.parameters()),
                        int(epoch),
                        int(config['data']['image_height']),
                        optimizer.param_groups[0]["lr"],
                        batch_size,
                        num_samples,
                        num_samples_val,
                        round(float(train_loss), 4), 
                        round(float(val_metrics["Loss"]), 4),
                        round(float(val_metrics["Loss_All"]), 4),
                        round(float(train_metrics["Dice"]), 4),
                        round(float(val_metrics["Dice"]), 4), 
                        round(duration, 2),
                        round(peak_vram, 3),
                        total_flops,
                        round(latency_ms, 3),
                        round(throughput, 2),
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

                if save_model and val_metrics['Dice'] > best_val_dice:
                    best_val_dice = val_metrics['Dice']
                    torch.save(model.state_dict(), checkpoint_path)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
