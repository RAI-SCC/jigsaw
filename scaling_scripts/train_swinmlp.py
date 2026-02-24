import argparse
import time
import yaml
import torch
from torch.profiler import ProfilerActivity
import os
import csv
import torch.optim as optim
import utils 
from perun import monitor

@monitor()
def train_one_epoch(
    model, 
    loader, 
    optimizer, 
    loss_fn, 
    device, 
    profiler = None
):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        # Load data to device
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
            loss = loss_fn(predictions, masks)

        # Backward Pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = utils.build_swin_model(config, device)
    train_loader, val_loader = utils.build_swin_dataloaders(config)

    num_epochs = config["training"]["epochs"]
    batch_size = config["training"]["train_batch"]
    learn_rate = config["training"]["lr"]
    save_dir = config["training"]["save_dir"]
    num_samples = len(train_loader.dataset)
    num_samples_val = len(val_loader.dataset)
    shifted_windows = config['SWIN_MLP']['shifted_windows']
    kernel_flex = config['SWIN_MLP']['kernel_flex']
    if kernel_flex:
        kernel_str = "flex"
    else:        
        kernel_str = "fixed"
    
    if shifted_windows:
        shift_str = "shift"
    else:
        shift_str = "nonshift"

    seed = config["SWIN_MLP"]["seed"]

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
        patience=3,    
        factor=0.5,   
        threshold=1e-3,
        min_lr=1e-6, 
    )

    loss_fn = utils.build_loss_fn(config)

    os.makedirs(save_dir, exist_ok=True)
    log_file = f"log_{config['data']['image_height']}_{num_epochs}_{learn_rate}_{kernel_str}_{shift_str}_{seed}.csv"
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

    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        img = batch[0].to(device)
        mask = batch[1].to(device)
        
        out = model(img)
        print(f"Output shape:    {out.shape}")
        print(f"Output min/max:  {out.min():.4f} / {out.max():.4f}")
        print(f"Mask unique:     {mask.unique()}")
        print(f"Mask shape:      {mask.shape}")
        
        probs = torch.softmax(out, dim=1)
        pred = probs.argmax(dim=1)
        print(f"Pred unique:     {pred.unique()}")
    model.train()

    # Calculate the number of batches per epoch to determine the profiling schedule
    batches_per_epoch = len(train_loader)
    wait_steps = max(0, batches_per_epoch - 6)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait_steps, warmup=3, active=3, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(save_dir),
        with_flops=True,
        record_shapes=True,
        with_modules=True,
        profile_memory=True,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:  
        
        for epoch in range(num_epochs):
            torch.cuda.reset_peak_memory_stats() 
            torch.cuda.synchronize()
            start_time = time.time()

            train_loss = train_one_epoch(
                model= model, loader=train_loader, optimizer=optimizer,
                loss_fn=loss_fn, device=device, profiler=prof)
           
            train_metrics = utils.evaluate(train_loader, model,loss_fn, device)
            val_metrics   = utils.evaluate(val_loader, model, loss_fn, device)
            
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

if __name__ == "__main__":
    main()
