import glob

import torch.utils.data as data
import torchvision
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import torchvision.transforms.functional as F



class MaskedCelebA(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[32, 32]):
        imgs = glob.glob(os.path.join(data_root, "image*"))
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = torch.from_numpy(np.load(path))
        mask = torch.from_numpy(np.load(path.replace("image", "mask").replace("png", "npy")))
        img = img * mask

        return {"image": img, "mask": mask}

    def __len__(self):
        return len(self.imgs)


class MRILoader(data.Dataset):
    def __init__(self, data_root, data_len=-1, mask=False):
        imgs = glob.glob(os.path.join(data_root, "image*"))
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.mask = mask

    def __getitem__(self, index):
        path = self.imgs[index]
        img = torch.from_numpy(np.load(path))
        if self.mask:
            mask = torch.from_numpy(np.load(path.replace("image", "mask").replace("png", "npy")))
        else:
            mask = torch.ones_like(img)
        img = img * mask
        # normalize
        img = img / 7e-5

        return {"image": img, "mask": mask}

    def __len__(self):
        return len(self.imgs)



from functools import reduce  # Required in Python 3
import operator


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebAWrapper(torchvision.datasets.CelebA):
    def __init__(self, data_root, image_size=[64, 64], rand_flip=False, grayscale=False, **kwargs):
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        self.tfs = transforms.Compose([
            Crop(x1, x2, y1, y2),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip() if rand_flip else lambda x: x,
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.data_root = data_root
        super(CelebAWrapper, self).__init__(root=data_root, transform=self.tfs, download=True, **kwargs)
        self.image_size = image_size
        self.grayscale = grayscale

    def __getitem__(self, index):
        img = super(CelebAWrapper, self).__getitem__(index)[0]
        if self.grayscale:
            img = img.mean(0, keepdim=True)
        return img


