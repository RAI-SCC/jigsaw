import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path 
import torchvision.transforms as transforms 
from perun import monitor

class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(512, 512), num_slices=1):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_slices = num_slices
        self.patient_images = self.get_patient_data(images_dir)
        self.patient_masks = self.get_patient_data(masks_dir)
        self.samples = []
        self.ids = [file for file in os.listdir(images_dir) if not file.startswith('.')]
        
        # Transform pictures into target size
        # Purpose: resize images so you dont have to reload the data each time
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor() 
        ])

        # Transform masks into target size
        # Purpose: resize masks so you dont have to reload the data each time
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # Generate samples of consecutive slices, store as (patient_id, start_slice_index)
        for pid in self.patient_images:
            images = self.patient_images[pid]
            num_available_slices = len(images)
            max_start_index = num_available_slices - self.num_slices + 1
            
            for start_slice_index in range(max_start_index):
                self.samples.append((pid, start_slice_index))

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
    @monitor()
    def __getitem__(self, index):
        pid, start_slice_index = self.samples[index]
        image_slices = []
        mask_slices = []

        for i in range(self.num_slices):
            slice_index = start_slice_index + i
            image_path = self.patient_images[pid][slice_index][1]
            mask_path = self.patient_masks[pid][slice_index][1]

            image = Image.open(image_path).convert('L')
            mask = Image.open(mask_path).convert('L')

            image = self.transform(image)
            image = (image - 0.5) / 0.5
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()

            image_slices.append(image)
            mask_slices.append(mask)

        image_tensor = torch.stack(image_slices).squeeze(1)
        mask_tensor = torch.stack(mask_slices).squeeze(1)

        return image_tensor, mask_tensor

def get_dataloader(config_dict, data_path, mode, distributed, batch_size, subset_size=None, rank=0):
    
    h = config_dict["data"]["image_height"]
    w = config_dict["data"]["image_width"]
    target_res = (h, w)
    
    images_dir = os.path.join(data_path, "image")
    masks_dir = os.path.join(data_path, "mask")
    
    if rank == 0:
        print(f"[{mode}] Creating Dataset from {images_dir}")

    dataset = BasicDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=target_res,
        num_slices=1
    )

    sampler = None
    if distributed:
        # Use DistributedSampler to split the dataset among multiple processes
        # for example GPU 0 gets the first 100 samples, GPU 1 the next 100 samples, etc.
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=rank,
            shuffle=(mode == "train"), 
            drop_last= (mode == "train")
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train" and sampler is None),
        sampler=sampler,
        num_workers=config_dict["data"]["num_workers"],
        pin_memory=True,
        drop_last=(mode == "train")
    )
