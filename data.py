from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import BILINEAR, LANCZOS
import cv2
import albumentations as A
from pathlib import Path
import numpy as np


class SegDataset(Dataset):

    def __init__(self,images_path, masks_path, labels, size, transform = None,
                 resample = BILINEAR):

        '''
        pass masks_path = None to return only x_batch
        '''
        super().__init__()
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path) if masks_path else None
        self.names = labels
        self.transform = transform
        self.size = size
        self.max_pixel_value = 255
        self.resample = resample

    def get_image_path(self,i):

        return self.images_path/f"{self.names[i]}.jpg"

    def get_mask_path(self, i):

        return self.masks_path/f"{self.names[i]}.npy"

    def load_image_and_mask(self,i):
        image_path = self.get_image_path(i)
        image = Image.open(image_path)
        image = image.resize(self.size, resample = self.resample)

        if self.masks_path:
            mask_path = self.get_mask_path(i)
            mask = np.load(mask_path)
            mask = cv2.resize(mask, self.size) ### no new classes after resize with default interpolation

            return np.array(image), mask

        else:
            return np.array(image)

    def __getitem__(self, i):

        item = self.load_image_and_mask(i)
        image, mask = item if self.masks_path else (item,None)
        if self.transform: ### augmentation from albumentation, which transforms both image and mask

            if self.masks_path:
                aug_pair = self.transform(image = image, mask = mask)
                image = aug_pair['image']
                mask = aug_pair['mask']
            else:
                aug_pair = self.transform(image = image)
                image = aug_pair['image']

        image = image / self.max_pixel_value
        image = np.transpose(image, (2,0,1)) ### H x W x C -> C x H x W

        if self.masks_path:
            return image.astype(np.float32), mask.astype(np.int64)

        else:
            return image.astype(np.float32)

    def __len__(self):
        return len(self.names)
