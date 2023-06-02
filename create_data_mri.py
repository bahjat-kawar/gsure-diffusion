import argparse

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import h5py
import sigpy as sp


def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis).to(tensor.device) # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

def get_mask(batch_size=1, acs_lines=30, total_lines=320, R=1):
    # Overall sampling budget
    num_sampled_lines = total_lines // R

    # Get locations of ACS lines
    # !!! Assumes k-space is even sized and centered, true for fastMRI
    center_line_idx = torch.arange((total_lines - acs_lines) // 2,
                                (total_lines + acs_lines) // 2)

    # Find remaining candidates
    outer_line_idx = torch.cat([torch.arange(0, (total_lines - acs_lines) // 2), torch.arange((total_lines + acs_lines) // 2, total_lines)])
    random_line_idx = shufflerow(outer_line_idx.unsqueeze(0).repeat([batch_size, 1]), 1)[:, : num_sampled_lines - acs_lines]
    # random_line_idx = outer_line_idx[torch.randperm(outer_line_idx.shape[0])[:num_sampled_lines - acs_lines]]

    # Create a mask and place ones at the right locations
    mask = torch.zeros((batch_size, total_lines))
    mask[:, center_line_idx] = 1.
    mask[torch.arange(batch_size).repeat_interleave(random_line_idx.shape[-1]), random_line_idx.reshape(-1)] = 1.

    return mask

class H5_Loader(Dataset):
    def __init__(self, file_list, input_dir,
                 project_dir='./',
                 R=1,
                 image_size=(320, 320),
                 acs_size=26,
                 pattern='random',
                 orientation='vertical'):
        # Attributes
        self.project_dir = project_dir
        self.file_list = file_list
        self.input_dir = input_dir
        self.image_size = image_size
        self.R = R
        self.pattern = pattern
        self.orientation = orientation

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list, )), dtype=int)
        for idx, file in enumerate(tqdm(self.file_list)):
            input_file = os.path.join(self.input_dir, os.path.basename(file))
            with h5py.File(os.path.join(self.project_dir, input_file), 'r') as data:
                self.num_slices[idx] = int(np.array(data['kspace'])[10:41].shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices))  # Total number of slices from all scans

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1 + wout, y1:y1 + hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)


        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(os.path.join(self.project_dir, raw_file), 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'])[10:41][slice_idx][None, :, :]
        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (
            gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0],
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,))  # Back to k-space

        gt_ksp = np.concatenate([np.real(gt_ksp), np.imag(gt_ksp)], axis=0)
        return gt_ksp

def create_train(output_clean, output_corrupt, R=4, sigma = 0.01):
    file_list = glob.glob(os.path.join("datasets/fmri/singlecoil_train", "*.h5"))
    input_dir = "datasets/fmri/singlecoil_train/"
    dataset = H5_Loader(file_list=file_list, input_dir=input_dir)
    train_loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        drop_last=False
    )
    indx = 0
    os.makedirs(output_clean, exist_ok=True)
    os.makedirs(output_corrupt, exist_ok=True)
    for images in tqdm(train_loader):
        noise = torch.randn_like(images) * sigma
        masks = get_mask(batch_size=4, R=R).unsqueeze(1).unsqueeze(1)
        images = images * 7e-5
        images_corrupt = images + noise
        images_corrupt = images_corrupt * masks
        masks = masks.repeat([1, 2, 320, 1])
        for image, image_corrupt, mask in zip(images.unbind(0), images_corrupt.unbind(0), masks.unbind(0)):
            np.save(os.path.join(output_clean, f"image_{indx:06d}"), image.cpu().numpy())
            np.save(os.path.join(output_corrupt, f"image_{indx:06d}"), image_corrupt.cpu().numpy())
            np.save(os.path.join(output_corrupt, f"mask_{indx:06d}"), mask.cpu().numpy())
            indx = indx + 1

    print(f"total {indx} datapoints")


def create_validation(output):
    file_list = glob.glob(os.path.join("datasets/fmri/singlecoil_val", "*.h5"))
    input_dir = "datasets/fmri/singlecoil_val/"
    dataset = H5_Loader(file_list=file_list, input_dir=input_dir)
    train_loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        drop_last=False
    )
    indx = 0
    os.makedirs(output, exist_ok=True)
    for images in tqdm(train_loader):
        for image in images.unbind(0):
            np.save(os.path.join(output, f"image_{indx:06d}"), image.cpu().numpy())
            indx = indx + 1

    print(f"total {indx} datapoints")

def main(output_dir, R, sigma):
    create_train(os.path.join(output_dir, "train_clean"), os.path.join(output_dir, f"train_corrupted"), R=R, sigma=sigma)
    create_validation(os.path.join(output_dir, "val"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='datasets/knee_mri_singlecoil/',
                        help='path for MRI dataset creation')
    parser.add_argument('-R', type=int, help='acceleration value for corrupt data creation')
    parser.add_argument('-s', '--sigma', type=float, help='sigma value for corrupt data creation')
    args = parser.parse_args()
    main(output_dir=args.output, R=args.R, sigma=args.sigma)
