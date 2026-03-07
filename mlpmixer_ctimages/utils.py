from loss import dice_coeff, CETopKDiceLoss
from Model import ModelMLPMixer
import Dataloaders
from Dataloaders import BasicDataset
import os
from typing import Dict, Tuple
import torch
from perun import monitor

def build_loss_fn(config_dict):
    """Return loss function."""
    loss_type = config_dict["training"]["loss"]
    match loss_type:
        case "TopKDice":
            loss_fn = CETopKDiceLoss(k=config_dict["training"]["top_k"])
        case "L1":
            loss_fn = torch.nn.L1Loss()
        case "L2":
            loss_fn = torch.nn.MSELoss()
        case _:
            raise ValueError("Only L1 and L2 losses are supported.")
    return loss_fn


def build_swin_model(
    config_dict: Dict,
    device: torch.device,
) -> torch.nn.Module:
    
    data_cfg = config_dict["data"]
    model_cfg = config_dict["model"]
    
    resolution = (data_cfg["image_height"], data_cfg["image_width"])

    model = ModelMLPMixer(
        channels_in=data_cfg["in_channels"],
        input_resolution=resolution,
        classes=data_cfg["num_classes"],
        mlp_ratio=model_cfg["mlp_ratio"],
        kernel_flex= model_cfg["kernel_flex"]
    ).to(device)

    return model


def build_swin_dataloaders(
        config_dict: dict
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_path = config_dict["data"]["train_data_path"]
    valid_path = config_dict["data"]["valid_data_path"]
    train_batch = config_dict["training"]["train_batch"]
    valid_batch = config_dict["training"]["valid_batch"]

    h = config_dict["data"]["image_height"]
    w = config_dict["data"]["image_width"]
    target_res = (h, w)

    train_dataset = BasicDataset(
        images_dir=os.path.join(train_path, "image"),
        masks_dir=os.path.join(train_path, "mask"),
        num_slices=1,
        is_train=True
    )

    valid_dataset = BasicDataset(
        images_dir=os.path.join(valid_path, "image"),
        masks_dir=os.path.join(valid_path, "mask"),
        num_slices=1,
        is_train=False
    )

    train_loader = Dataloaders.get_dataloader(
        config_dict=config_dict,
        dataset=train_dataset,
        mode="train",
        batch_size=train_batch,
    )

    valid_loader = Dataloaders.get_dataloader(
        config_dict=config_dict,
        dataset=valid_dataset,
        mode="val",
        batch_size=valid_batch,
    )

    return train_loader, valid_loader

@monitor()
@torch.no_grad()
def evaluate(loader, model, loss_fn, loss_fn_all, device, threshold=0.5):
    model.eval()

    running_loss = 0.0
    running_loss_all = 0.0
    total_samples = 0
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0

    TP = FP = FN = TN = 0

    for batch in loader:
        images = batch[0].to(device)
        true_masks = batch[1].to(device) 

        if true_masks.dim() == 4 and true_masks.shape[1] == 1:
            true_masks = true_masks.squeeze(1) 

        true_masks = (true_masks > 0.5).float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images) 

        logits_f32 = logits.float()
        loss = loss_fn(logits_f32, true_masks)
        loss_all = loss_fn_all(logits_f32, true_masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size                 
        running_loss_all += loss_all.item() * batch_size
        total_samples += batch_size

        # Probabilities
        probs = torch.softmax(logits_f32, dim=1)[:, 1]
        preds = (probs > threshold).float()

        # Accuracy
        num_correct += (preds == true_masks).sum().item()
        num_pixels += torch.numel(preds)

        # Confusion
        TP += ((preds == 1) & (true_masks == 1)).sum().item()
        FP += ((preds == 1) & (true_masks == 0)).sum().item()
        FN += ((preds == 0) & (true_masks == 1)).sum().item()
        TN += ((preds == 0) & (true_masks == 0)).sum().item()

        # Dice
        batch_dice = dice_coeff(
            preds.unsqueeze(1),
            true_masks.unsqueeze(1),
            reduce_batch_first=False
        )
        dice_score += batch_dice.item() * batch_size
        
    avg_loss = running_loss / total_samples
    avg_loss_all = running_loss_all / total_samples

    acc = num_correct / num_pixels * 100 if num_pixels > 0 else 0
    avg_dice = dice_score / total_samples
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)

    return {
        "Loss": avg_loss,
        "Loss_All": avg_loss_all,
        "Accuracy": acc,
        "Dice": avg_dice,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": precision,
        "Recall": recall,
    }