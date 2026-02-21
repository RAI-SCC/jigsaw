import torch
from torch import Tensor
import torch.nn as nn

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
    def __init__(self, k: float = 10, smooth: float = 1e-5):
        super(TopKDiceLoss, self).__init__()
        self.k = k
        self.smooth = smooth
        
    # logits: (Batch, 2, H, W) -> From Modell
    # target: (Batch, 1, H, W) -> From DataLoader    
    def forward(self, logits: Tensor, target: Tensor):
        
        # Calculate probabilities
        probs = torch.softmax(logits, dim=1)
        probs = probs[:, 1, ...]  # Assuming class 1 is the foreground class
        
        # Ensure the dimensions match
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        elif target.dim() == 3:
            pass

        assert probs.shape == target.shape, f"Shapes dont match: {probs.shape} vs {target.shape}"
            
        # Flatten everything to work on a pixel-wise basis
        probs_flat = probs.reshape(probs.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Compute "True Positive Map"
        tp_map = probs_flat * target_flat
        
        mask = torch.ones_like(tp_map)
        
        for i in range(probs.shape[0]):
            foreground_indices = (target_flat[i] == 1)
            
            if foreground_indices.sum() > 0:
                foreground_tp = tp_map[i][foreground_indices]

                k_num = int(foreground_tp.numel() * (self.k / 100.0))
                k_num = max(1, k_num) 
                
                threshold_val, _ = torch.kthvalue(foreground_tp, k_num)
                
                # Create "Ignore Map"
                pixel_to_ignore = (target_flat[i] == 1) & (tp_map[i] > threshold_val)
                mask[i][pixel_to_ignore] = 0.0

        probs_k = probs_flat * mask
        target_k = target_flat * mask

        intersection = (probs_k * target_k).sum(dim=1)
        union = probs_k.sum(dim=1) + target_k.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()