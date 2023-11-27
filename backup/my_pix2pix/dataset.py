import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset
# from torchvision.utils import save_image
import nibabel as nib
import nibabel.processing as nibproc
import torch


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.list_files = self.list_files[config.LEFT : config.RIGHT+1];
        self.source_images = []
        self.target_images = []

        for index in range(len(self.list_files)):
            id = self.list_files[index]
            src_img_path = self.root_dir + id + "/mr.nii.gz"
            target_img_path = self.root_dir + id + "/ct.nii.gz"
            
            nii_img = nib.load(src_img_path)
            nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
            input_image = np.array(nii_img.get_fdata())
            
            nii_img = nib.load(target_img_path)
            nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
            target_image = np.array(nii_img.get_fdata())

            for i in range(50, 150, 3):
                sliced_input_image = input_image[:, :, i:i+3]
                sliced_target_image = target_image[:, :, i:i+3]
                sliced_input_image = sliced_input_image.transpose(2, 0, 1)
                sliced_target_image = sliced_target_image.transpose(2, 0, 1)

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, index):
        return self.source_images[index], self.target_images[index]
    
# class MapDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.list_files = os.listdir(self.root_dir)
#         self.list_files = self.list_files[config.LEFT : config.RIGHT+1];

#     def __len__(self):
#         return len(self.list_files)

#     def __getitem__(self, index):
#         id = self.list_files[index]
#         src_img_path = self.root_dir + id + "/mr.nii.gz"
#         target_img_path = self.root_dir + id + "/ct.nii.gz"
        
#         nii_img = nib.load(src_img_path)
#         nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
#         input_image = np.array(nii_img.get_fdata())
        
#         nii_img = nib.load(target_img_path)
#         nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
#         target_image = np.array(nii_img.get_fdata())

#         input_image = input_image[:, :, 100]
#         target_image = target_image[:, :, 100]
#         # input_image = input_image.transpose(2, 0, 1)
#         # target_image = target_image.transpose(2, 0, 1)
#         # input_image = torch.from_numpy(input_image)
#         # target_image =  torch.from_numpy(target_image)

#         input_image = config.transform_only_mask(image=input_image)["image"]
#         target_image = config.transform_only_mask(image=target_image)["image"]

#         # print(input_image.shape, target_image.shape)
#         return input_image, target_image

if __name__ == "__main__":
    dataset = MapDataset("Task1/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        # print(x.shape)
        # save_image(x, "x.png")
        # save_image(y, "y.png")
        import sys

        sys.exit()