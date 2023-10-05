import os
from torch.utils.data import Dataset
from PIL import Image

class CustomSegmentationDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'masks')

        # List of image and mask filenames
        self.image_filenames = os.listdir(self.image_dir)
        self.mask_filenames = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load an image and its corresponding mask
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            # Apply augmentations to both image and mask
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)

        return {'image': image, 'mask': mask}
