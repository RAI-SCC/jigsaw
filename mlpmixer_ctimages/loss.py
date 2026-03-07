import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculates the Dice coefficient between two binary masks
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target, reduce_batch_first=True)


class TopKDiceLoss(nn.Module):
    def __init__(self, k: float = 50, smooth: float = 1e-5):
        super(TopKDiceLoss, self).__init__()
        self.k = k
        self.smooth = smooth
        self._epsilon = None  # Buffer to hold epsilon values for tie-breaking in top K selection
        
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

        # init epsilon once
        # added small random noise to epsilon to prevent ties in top K selection
        if self._epsilon is None or self._epsilon.shape != probs_flat.shape:
             self._epsilon = torch.rand_like(probs_flat) * 1e-6

        # Compute "True Positive Map"
        tp_map = probs_flat * (target_flat + self._epsilon.to(probs_flat.device))  # Add small noise to target to break ties
        
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
        sum_sq_pred = (probs_k ** 2).sum(dim=1)
        sum_sq_target = (target_k ** 2).sum(dim=1)
        union = sum_sq_pred + sum_sq_target
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        topk_dice_loss = 1 - dice.mean()

        return topk_dice_loss
    
class CETopKDiceLoss(nn.Module):
    def __init__(self, k: float = 50, smooth: float = 1e-5):
        super(CETopKDiceLoss, self).__init__()
        self.topk_dice = TopKDiceLoss(k=k, smooth=smooth)
   
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:

        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        ce_loss = F.cross_entropy(logits, target.long())

        topk_dice_loss = self.topk_dice(logits, target)

        return ce_loss + topk_dice_loss