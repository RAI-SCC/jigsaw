import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(1024, 1024), num_slices=1, is_train=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_slices = num_slices
        self.patient_images = self.get_patient_data(images_dir)
        self.patient_masks = self.get_patient_data(masks_dir)
        self.samples = []
        self.ids = [file for file in os.listdir(images_dir) if not file.startswith('.')]
        
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

        for pid in self.patient_images:
            images = self.patient_images[pid]
            num_available_slices = len(images)
            for slice_index in range(num_available_slices):
                self.samples.append((pid, slice_index))

    # create dictionary with patient ids as keys and list of (slice_num, path) tuples as values
    def get_patient_data(self, images_dir):
        all_image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
        patient_dict = {}

        for images in all_image_files:
            base_name = os.path.basename(images)
            pid = base_name.rsplit('_', 1)[0]
            slice_num = int(base_name.rsplit('_', 1)[1].split('.')[0])
            
            if pid not in patient_dict:
                patient_dict[pid] = [] 
            patient_dict[pid].append((slice_num, images))
        
        for pid in patient_dict:
            patient_dict[pid].sort()        
        return patient_dict

    def __len__(self):
        return len(self.samples)

    # Retrieve a sample consisting of consecutive slices for a given index
    def __getitem__(self, index):
        pid, slice_index = self.samples[index]
        img_path = self.patient_images[pid][slice_index][1]
        mask_path = self.patient_masks[pid][slice_index][1]
        
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']

        mask_tensor = (mask_tensor.float() / 255.0 > 0.5).float()

        return image_tensor, mask_tensor

def get_dataloader(config_dict, dataset, mode, batch_size, subset_size=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=config_dict["data"]["num_workers"],
        pin_memory=True,
        drop_last=(mode == "train")
    )
