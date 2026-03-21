import os
import glob
import random
import cv2
import numpy as np
import yaml
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    IMAGE_DIR = config['data']['image_dir']
    MASK_DIR = config['data']['mask_dir']
    BASE_DIR = config['data']['base_dir']
    SEED = config['SWIN_MLP']['seed']

    H = config['data']['image_height']
    W = config['data']['image_width']
    TARGET_SIZE = (W, H) 
    size = str(H)

    all_image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

    patient_ids = set()

    for image_path in all_image_files:
        base_name = os.path.basename(image_path)
        patient_id = base_name.rsplit('_', 1)[0] 
        patient_ids.add(patient_id)

    BLACKLISTED_IDS = {
        "ID00035637202182204917484",
        "ID00014637202177757139317",
        "ID00032637202181710233084",
        "ID00161637202235731948764",
        "ID00123637202217151272140",
        "ID00027637202179689871102",
        "ID00075637202198610425520",
        "ID00139637202231703564336",
        "ID00426637202313170790466"
    }

    patient_ids = patient_ids - BLACKLISTED_IDS
    patient_ids = sorted(list(patient_ids))
    random.seed(SEED)
    random.shuffle(patient_ids)

    num_total = len(patient_ids)
    train_split_end = int(num_total * 0.7)
    val_split_end = int(num_total * 0.8)

    train_ids = set(patient_ids[:train_split_end])
    val_ids = set(patient_ids[train_split_end:val_split_end])
    test_ids = set(patient_ids[val_split_end:])

    split_info = {
        "seed": SEED,
        "total_patients": num_total,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "train_ids": sorted(list(train_ids)),
        "val_ids": sorted(list(val_ids)),
        "test_ids": sorted(list(test_ids))
    }

    print(f'Total: {num_total} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')

    splits = ['train', 'val', 'test']
    types = ['image', 'mask']

    for split in splits:
        for type_ in types:
            path = os.path.join(BASE_DIR, size, split, type_)
            os.makedirs(path, exist_ok=True)
            print(f"Created/checked directory: {path}")

    info_file_path = os.path.join(BASE_DIR, size, 'split_info.json')
    with open(info_file_path, "w") as f:
        json.dump(split_info, f, indent=4)

    print(f'Total patients: {num_total}')
    print(f'Training patients: {len(train_ids)}')
    print(f'Validation patients: {len(val_ids)}')
    print(f'Testing patients: {len(test_ids)}')


    for image_path in all_image_files:
        base_name = os.path.basename(image_path)
        patient_id = base_name.rsplit('_', 1)[0] 

        try:
            slice_part = base_name.rsplit('_', 1)[1] 
        except IndexError:
            print(f"Invalid format skipped: {base_name}")
            continue

        mask_base_name = f"{patient_id}_mask_{slice_part}" 
        mask_path = os.path.join(MASK_DIR, mask_base_name)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {base_name}")
            continue

        if patient_id in train_ids:
            split_dir = 'train'
        elif patient_id in val_ids:
            split_dir = 'val'
        elif patient_id in test_ids:
            split_dir = 'test'
        else:
            continue

        # load image as grayscale     
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        if img.shape[:2] == (H, W):
            img_resized = img
        else:
            img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

        # load and process mask
        multiclass_mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR) 
        
        if multiclass_mask_color is None: continue

        lower_blue = np.array([190, 0, 0], dtype=np.uint8)
        upper_blue = np.array([255, 60, 60], dtype=np.uint8)
        binary_mask = cv2.inRange(multiclass_mask_color, lower_blue, upper_blue)

        if binary_mask.shape[:2] == TARGET_SIZE:
            mask_resized = binary_mask
        else:
            mask_resized = cv2.resize(binary_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(BASE_DIR, size, split_dir, 'image', base_name), img_resized)
        cv2.imwrite(os.path.join(BASE_DIR, size, split_dir, 'mask', base_name), mask_resized)

    print("Processing completed!")

if __name__ == "__main__":
    main()