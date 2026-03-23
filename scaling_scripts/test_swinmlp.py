import os
import random
import argparse
import yaml
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import DataLoader

from src.Dataloaders import BasicDataset
from src.Model import ModelSwinMLP
from src.dice import DiceLoss

COLOR_TP = np.array([0,   150, 130]) / 255   
COLOR_FP = np.array([223, 155,  27]) / 255   
COLOR_FN = np.array([162,  34,  35]) / 255   


def load_model(config, checkpoint_path, device):
    data_cfg = config["data"]
    swin_cfg = config["SWIN_MLP"]

    model = ModelSwinMLP(
        channels_in=data_cfg["in_channels"],
        input_resolution=(data_cfg["image_height"], data_cfg["image_width"]),
        classes=data_cfg["num_classes"],
        window_size=swin_cfg["window_size"],
        mlp_ratio=swin_cfg["mlp_ratio"],
        shifted_windows=swin_cfg["shifted_windows"],
        kernel_flex=swin_cfg["kernel_flex"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both direct state_dict and wrapped format
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    
    model.load_state_dict(ckpt)
    model.eval()
    return model


@torch.no_grad()
def run_evaluation_per_image(model, dataset, device, threshold=0.5):
    dice_fn = DiceLoss()
    per_image_metrics = []
    
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        image = image.unsqueeze(0).to(device)
        mask = mask.to(device)
        
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        elif mask.dim() == 2:
            pass 
        
        mask = (mask > 0.5).float()
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image)
        
        logits_f32 = logits.float()
        probs = torch.softmax(logits_f32, dim=1)[:, 1].squeeze(0) 
        preds = (probs > threshold).float()
        
        TP = ((preds == 1) & (mask == 1)).sum().item()
        FP = ((preds == 1) & (mask == 0)).sum().item()
        FN = ((preds == 0) & (mask == 1)).sum().item()
        TN = ((preds == 0) & (mask == 0)).sum().item()
        
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-6) * 100
        
        image_dice = 1 - dice_fn(logits_f32, mask.unsqueeze(0))
        
        per_image_metrics.append({
            "image_idx": idx,
            "dice": image_dice.item(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
        })

    return per_image_metrics

@torch.no_grad()
def run_evaluation_aggregate(model, loader, device, threshold=0.5):
    dice_fn = DiceLoss()
    TP = FP = FN = TN = 0
    dice_sum = 0.0
    total = 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        masks = (masks > 0.5).float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)

        logits_f32 = logits.float()
        probs = torch.softmax(logits_f32, dim=1)[:, 1]
        preds = (probs > threshold).float()

        TP += ((preds == 1) & (masks == 1)).sum().item()
        FP += ((preds == 1) & (masks == 0)).sum().item()
        FN += ((preds == 0) & (masks == 1)).sum().item()
        TN += ((preds == 0) & (masks == 0)).sum().item()

        batch_dice = 1 - dice_fn(logits_f32, masks)
        dice_sum  += batch_dice.item() * images.size(0)
        total     += images.size(0)

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    dice      = dice_sum / total
    accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-6) * 100

    return {
        "Dice":      dice,
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
        "Accuracy":  accuracy,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }


def denormalize(tensor):
    arr = tensor.squeeze().cpu().numpy()
    arr = arr * 0.5 + 0.5
    return np.clip(arr, 0, 1)


def make_overlay(image_np, pred_np, mask_np):
    h, w   = image_np.shape
    rgb    = np.stack([image_np] * 3, axis=-1)  
    overlay = rgb.copy()

    tp_mask = (pred_np == 1) & (mask_np == 1)
    fp_mask = (pred_np == 1) & (mask_np == 0)
    fn_mask = (pred_np == 0) & (mask_np == 1)

    alpha = 0.55
    for color, region in [(COLOR_TP, tp_mask),
                           (COLOR_FP, fp_mask),
                           (COLOR_FN, fn_mask)]:
        overlay[region] = (1 - alpha) * rgb[region] + alpha * color

    return overlay


@torch.no_grad()
def visualize_samples(model, dataset, device, indices, threshold=0.5, 
                      output_file="test_visual.png"):
    n_samples = len(indices)
    n_cols  = 4   # image | ground truth | prediction | overlay
    n_rows  = n_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "Prediction", "Overlay (TP/FP/FN)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_in = image.unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(image_in)

        probs = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
        pred  = (probs > threshold).astype(np.float32)

        image_np = denormalize(image)
        mask_np  = mask.numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        mask_np  = (mask_np > 0.5).astype(np.float32)

        overlay = make_overlay(image_np, pred, mask_np)

        tp = int(((pred == 1) & (mask_np == 1)).sum())
        fp = int(((pred == 1) & (mask_np == 0)).sum())
        fn = int(((pred == 0) & (mask_np == 1)).sum())
        img_dice = 2 * tp / (2 * tp + fp + fn + 1e-6)

        axes[row, 0].imshow(image_np,  cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(mask_np,   cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred,       cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(overlay)

        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=9)
        axes[row, 2].set_xlabel(f"Dice: {img_dice:.3f}", fontsize=9,
                                 color="#404040", fontweight="bold")

        for ax in axes[row]:
            ax.axis("off")

    patches = [
        mpatches.Patch(color=COLOR_TP, label="TP"),
        mpatches.Patch(color=COLOR_FP, label="FP"),
        mpatches.Patch(color=COLOR_FN, label="FN"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Test Set — Segmentation Predictions vs Ground Truth",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved: {output_file}")


def get_consistent_sample_indices(dataset_size, n_samples, seed):
    rng = np.random.RandomState(seed)
    indices = rng.choice(dataset_size, size=min(n_samples, dataset_size), replace=False)
    return sorted(indices.tolist())


def save_per_image_metrics(per_image_metrics, output_path, config_info):
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        
        # Header
        writer.writerow([
            "Model", "Resolution", "Kernel", "Image_Idx",
            "Dice", "Precision", "Recall", "F1", "Accuracy",
            "TP", "FP", "FN", "TN"
        ])
        
        # Data rows
        for metrics in per_image_metrics:
            writer.writerow([
                "SwinMLP",
                config_info["resolution"],
                config_info["kernel"],
                metrics["image_idx"],
                round(metrics["dice"], 6),
                round(metrics["precision"], 6),
                round(metrics["recall"], 6),
                round(metrics["f1"], 6),
                round(metrics["accuracy"], 4),
                metrics["TP"],
                metrics["FP"],
                metrics["FN"],
                metrics["TN"],
            ])


def save_aggregate_metrics(aggregate_metrics, output_path, config_info, sample_indices, seed, threshold):
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        
        writer.writerow([
            "Model", "Checkpoint", "Resolution", "Kernel", "Seed", "Threshold",
            "Dice", "Precision", "Recall", "F1", "Accuracy",
            "TP", "FP", "FN", "TN", "Sample_Indices"
        ])
        
        writer.writerow([
            "SwinMLP",
            config_info["checkpoint"],
            config_info["resolution"],
            config_info["kernel"],
            seed,
            threshold,
            round(aggregate_metrics["Dice"], 4),
            round(aggregate_metrics["Precision"], 4),
            round(aggregate_metrics["Recall"], 4),
            round(aggregate_metrics["F1"], 4),
            round(aggregate_metrics["Accuracy"], 2),
            aggregate_metrics["TP"],
            aggregate_metrics["FP"],
            aggregate_metrics["FN"],
            aggregate_metrics["TN"],
            str(sample_indices)
        ])


def main():
    parser = argparse.ArgumentParser(description="Test SwinMLP segmentation model")
    parser.add_argument("--config", required=True,
                        help="Path to config.yaml")
    parser.add_argument("--n_visual", type=int, default=6,
                        help="Number of random samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducible sample selection (for visualization only)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get paths from config
    checkpoint_path = config["SWIN_MLP"]["checkpoint_file"]
    test_images_dir = config["data"]["test_data_path"]
    output_dir = config["testing"]["output_dir"]
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"\n\nCheckpoint not found: {checkpoint_path}\n")

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(config, checkpoint_path, device)

    test_dataset = BasicDataset(
        images_dir=os.path.join(test_images_dir, "image"),
        masks_dir=os.path.join(test_images_dir, "mask"),
        num_slices=1,
        is_train=False,
    )

    res = config["data"]["image_height"]
    kernel = "flex" if config["SWIN_MLP"]["kernel_flex"] else "fixed"
    
    config_info = {
        "checkpoint": checkpoint_path,
        "resolution": res,
        "kernel": kernel,
    }
    
    per_image_metrics = run_evaluation_per_image(
        model, test_dataset, device, threshold=args.threshold
    )
    
    per_image_csv = os.path.join(output_dir, f"per_image_metrics_{res}_{kernel}.csv")
    save_per_image_metrics(per_image_metrics, per_image_csv, config_info)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["valid_batch"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    aggregate_metrics = run_evaluation_aggregate(
        model, test_loader, device, threshold=args.threshold
    )
    
    sample_indices = get_consistent_sample_indices(
        len(test_dataset), 
        args.n_visual, 
        args.seed
    )

    vis_out = os.path.join(output_dir, f"test_visual_{res}_{kernel}.png")
    visualize_samples(
        model=model,
        dataset=test_dataset,
        device=device,
        indices=sample_indices,
        threshold=args.threshold,
        output_file=vis_out,
    )

    aggregate_csv = os.path.join(output_dir, f"aggregate_metrics_{res}_{kernel}.csv")
    save_aggregate_metrics(
        aggregate_metrics, aggregate_csv, config_info, 
        sample_indices, args.seed, args.threshold
    )

if __name__ == "__main__":
    main()