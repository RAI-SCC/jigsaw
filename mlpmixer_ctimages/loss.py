import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class TopKDiceLoss(nn.Module):
    def __init__(self, k: float = 50):
        super(TopKDiceLoss, self).__init__()
        self.k = k
        
    # logits: (Batch, 2, H, W) -> From model (background and foreground class logits)
    # target: (Batch, 1, H, W) -> From DataLoader, values 0 or 1 (binary mask)   
    def forward(self, logits: Tensor, target: Tensor):
        logits = logits.float()
        target = target.float()
        
        # Calculate probabilities, Softmax -> foreground probability map
        probs = torch.softmax(logits, dim=1)
        probs = probs[:, 1, ...]  # foreground class probability map (Batch, H, W)
        
        # Ensure the dimensions match
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1) #(B, 1, H, W) -> (B, H, W)
          
        # Flatten everything to work on a pixel-wise basis
        probs_flat = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        epsilon  = torch.rand_like(probs_flat) * 1e-6

        # Compute "True Positive Map"
        tp_map = probs_flat * (target_flat + epsilon)  # Add small noise to target to break ties
        
        # .detch() to seperate the "Ignore Map" from the computational graph,
        # so it doesn't affect gradients
        mask = torch.ones_like(tp_map).detach()
        
        # Determine the threshold for top K% pixels in the foreground class
        # use torch.no_grad() to ensure this doesn't affect the computational graph
        with torch.no_grad():
            for i in range(probs.shape[0]):
                foreground_indices = (target_flat[i] == 1)
                
                if foreground_indices.sum() == 0:
                    continue # No foreground pixels

                foreground_tp = tp_map[i][foreground_indices]

                # Determine the number of pixels to keep based on K%
                k_num = int(foreground_tp.numel() * (self.k / 100.0))
                k_num = max(1, k_num) 
                
                threshold_val, _ = torch.kthvalue(foreground_tp, k_num)
                
                # Create "Ignore Map"
                pixel_to_ignore = (target_flat[i] == 1) & (tp_map[i] > threshold_val)
                mask[i][pixel_to_ignore] = 0.0

        probs_k = probs_flat * mask
        target_k = target_flat * mask

        intersection = (probs_k * target_k).sum(dim=1)
        
        union        = probs_k.sum(dim=1) + target_k.sum(dim=1)
        union_safe = union.clamp(min=1e-6)
        dice = torch.where(
                union == 0,
                torch.ones_like(intersection), 
                (2. * intersection) / union_safe)
        
        topk_dice_loss = 1 - dice.mean()

        return topk_dice_loss
    
class CETopKDiceLoss(nn.Module):
    def __init__(self, k: float = 50):
        super(CETopKDiceLoss, self).__init__()
        self.topk_dice = TopKDiceLoss(k=k)
   
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:

        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        ce_loss = F.cross_entropy(logits, target.long())

        topk_dice_loss = self.topk_dice(logits, target)

        return 0.2 * ce_loss + topk_dice_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        logits = logits.float()
        target = target.float()

        probs = torch.softmax(logits, dim=1)
        probs = probs[:, 1, ...]

        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        probs_flat  = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        intersection  = (probs_flat * target_flat).sum(dim=1)
        union         = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        union_safe = union.clamp(min=1e-6)
        dice = torch.where(
                union == 0,
                torch.ones_like(intersection), 
                (2. * intersection) / union_safe)
        return 1 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super(CEDiceLoss, self).__init__()
        self.dice      = DiceLoss()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        ce_loss   = F.cross_entropy(logits, target.long())
        dice_loss = self.dice(logits, target)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss