import argparse
import time
import yaml
import torch
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
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        epoch_loss += loss.item()

        if rank == 0 and i % 10 == 0:
            print(f"Epoch [{epoch}][{i}/{len(loader)}] Loss: {loss.item():.4f}")

    return epoch_loss / len(loader)

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
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        avg_loss = train_one_epoch(
            config, epoch, model, train_loader, optimizer, loss_fn, device, rank
        )
        # wait for all processes to finish epoch before validating
        dist.barrier() 
        
        val_dice = utils.check_accuracy(val_loader, model, device, rank)
        dice_tensor = torch.tensor(val_dice).to(device)
        dist.broadcast(dice_tensor, src=0)
        scheduler.step(dice_tensor.item())
        
        # Learning Rate Step
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Dice: {val_dice:.4f} | LR: {current_lr:.1e} | Time: {(time.time()-start_time)/60:.1f} min")


            utils.save_predictions_as_imgs(
                val_loader, model, folder=f"{config['training']['save_dir']}/images", 
                device=device, epoch=epoch, rank=rank
            )

            utils.save_predictions_as_imgs(
                val_loader, model, folder=f"{config['training']['save_dir']}/images", 
                device=device, epoch=epoch, rank=rank
            )
            utils.save_model(
                model,
                optimizer,
                scheduler,
                config["training"]["save_dir"],
                rank,
                epoch,
            )
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()