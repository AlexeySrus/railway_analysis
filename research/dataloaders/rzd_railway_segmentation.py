from typing import Tuple, Optional

import albumentations as A
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_utils import create_square_crop_by_detection


def get_augmentations():
    transforms = A.Compose(
        [
            A.CoarseDropout(
                max_holes=12,
                max_height=256,
                max_width=256,
                min_holes=12,
                min_height=256,
                min_width=256,
                fill_value=0,
                mask_fill_value=0,
                p=0.8
            ),
            A.Perspective(scale=(0.1, 0.1), p=0.5, pad_val=114, mask_pad_val=0, interpolation=cv2.INTER_AREA),
            A.Affine(scale=(0.9, 1.5), interpolation=cv2.INTER_AREA, cval=(114, 114, 114), cval_mask=0, p=0.5, mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.25), contrast_limit=(0.25), p=0.5),
            A.MultiplicativeNoise(multiplier=(1.75, 1.75), per_channel=True, p=0.5),

         ]
    )
    return transforms


class RailwaySegmentation(Dataset):
    def __init__(self, root_path: str, size: int, apply_augmentations: bool = False):
        images_folder = os.path.join(root_path, 'images/')
        masks_folder = os.path.join(root_path, 'mask/')

        self.data = []

        for image_name in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_name)
            mask_path = os.path.join(masks_folder, image_name)

            self.data.append(
                {
                    'image': image_path,
                    'mask': mask_path
                }
            )

        self.size = size
        self.augmentations = None
        if apply_augmentations:
            self.augmentations = get_augmentations()

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return 4

    def read_image(self, img_path: str) -> np.ndarray:
        _img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert _img is not None, img_path
        return cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

    def read_mask(self, img_path: str) -> np.ndarray:
        _img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert _img is not None, img_path
        return _img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]

        image = self.read_image(sample['image'])
        raw_mask = self.read_mask(sample['mask'])

        assert image.shape[0] == raw_mask.shape[0] and image.shape[1] == image.shape[1]

        image = create_square_crop_by_detection(
            image,
            [0, 0, *image.shape[:2][::-1]],
            pad_value=114
        )

        raw_mask = create_square_crop_by_detection(
            raw_mask,
            [0, 0, *raw_mask.shape[:2][::-1]],
            pad_value=0
        )

        if self.augmentations is not None:
            aug_res = self.augmentations(image=image, mask=raw_mask)
            image = aug_res['image']
            raw_mask = aug_res['mask']

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        raw_mask = cv2.resize(raw_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        # image_tensor = image_tensor - torch.tensor([0.36436555, 0.36408448, 0.37672771]).unsqueeze(1).unsqueeze(1)
        # image_tensor = image_tensor / torch.tensor([0.17757423, 0.17931685, 0.20289507]).unsqueeze(1).unsqueeze(1)

        image_tensor = image_tensor - torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
        image_tensor = image_tensor / torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)

        mask = torch.zeros((self.size, self.size), dtype=torch.long)

        mask[raw_mask == 7] = 1
        mask[raw_mask == 6] = 2
        mask[raw_mask == 10] = 3

        return image_tensor, mask


if __name__ == '__main__':
    from PIL import Image

    dataset = RailwaySegmentation(
        root_path='/media/alexey/HDDData/datasets/railway/RZD_Segmentation/train/',
        size=1024,
        apply_augmentations=True
    )
    img, mask = dataset[5]

    img = img * torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)

    img = (torch.clip(img, 0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    mask = mask.numpy()

    class_colors = np.array(
        [
            [200, 50, 20],
            [10, 200, 20],
            [15, 200, 200]
        ]
    ).astype(np.float16)

    viz_img = img.copy()

    for i in range(3):
        cls_num = i + 1

        viz_img[mask == cls_num] = (
                img[mask == cls_num].astype(np.float16) * 0.4 +
                class_colors[i] * 0.6
        ).astype(np.uint8)

    Image.fromarray(viz_img).show()
