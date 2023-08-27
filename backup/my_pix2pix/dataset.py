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

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        id = self.list_files[index]
        src_img_path = self.root_dir + id + "/mr.nii.gz"
        target_img_path = self.root_dir + id + "/ct.nii.gz"
        
        nii_img = nib.load(src_img_path)
        nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
        input_image = np.array(nii_img.get_fdata())
        
        nii_img = nib.load(target_img_path)
        nii_img = nibproc.resample_from_to(nii_img, [(256, 256, 160), nii_img.affine])
        target_image = np.array(nii_img.get_fdata())

        input_image = input_image[:, :, 100:103]
        target_image = target_image[:, :, 100:103]
        input_image = input_image.transpose(2, 0, 1)
        target_image = target_image.transpose(2, 0, 1)
        # input_image = torch.from_numpy(input_image)
        # target_image =  torch.from_numpy(target_image)

        # print(input_image.shape, target_image.shape)
        return input_image, target_image

# class MapDataset(IterableDataset):
#     def __init__(self, root_dir, slice_thickness=3, start_slice=0, end_slice=159):
#         self.root_dir = root_dir
#         self.list_files = os.listdir(self.root_dir)
#         self.slice_thickness = slice_thickness
#         self.start_slice = start_slice
#         self.end_slice = end_slice

#     def __iter__(self):
#         for filename in os.listdir(self.data_dir):
#             image = torch.from_numpy(nib.load(os.path.join(self.data_dir, filename)).get_data())
#             for i in range(0, image.shape[1], self.slice_thickness):
#                 slices = image[:, i:i+self.slice_thickness, :]
#                 yield slices

#     def __len__(self):
#         return len(os.listdir(self.data_dir))

if __name__ == "__main__":
    dataset = MapDataset("Task1/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        # print(x.shape)
        # save_image(x, "x.png")
        # save_image(y, "y.png")
        import sys

        sys.exit()