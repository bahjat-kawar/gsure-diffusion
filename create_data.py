import argparse

from torch.utils import data
from torchvision.utils import save_image

from data.dataset import CelebAWrapper
import os
import numpy as np
import torch
from tqdm import tqdm



def create_train(output_corrupt, prob=0.2, sigma=0.01, patch=8):
    dataset = CelebAWrapper(data_root="datasets/", split="train", image_size=[32, 32], grayscale=True)
    train_loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        drop_last=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    indx = 0
    os.makedirs(output_corrupt, exist_ok=True)
    for images in tqdm(train_loader):
        images = images.to(device)
        images = images.mean(0, keepdim=True)
        b, c, h, w = images.shape
        masks = (torch.rand((b, c, patch, patch), device=device) > prob).int()
        masks = masks.repeat_interleave(h // patch, 2).repeat_interleave(w // patch, 3)
        noise = torch.randn_like(images) * sigma
        images_corrupt = images + noise
        images_corrupt = images_corrupt * masks
        for image_corrupt, mask in zip(images_corrupt.unbind(0), masks.unbind(0)):
            np.save(os.path.join(output_corrupt, f"image_{indx:06d}"), image_corrupt.cpu().numpy())
            np.save(os.path.join(output_corrupt, f"mask_{indx:06d}"), mask.cpu().numpy())
            indx = indx + 1

    print(f"total {indx} datapoints")


def main(output, p, sigma):
    create_train(output, prob=p, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='datasets/celeba_corrupted_prob02_sigma001/',
                        help='path for corrupt CelebA dataset creation')
    parser.add_argument('-p', type=int, default=0.2, help='probability for missing patch in corrupt data creation')
    parser.add_argument('-s', '--sigma', type=float, default=0.01, help='sigma value for corrupt data creation')
    args = parser.parse_args()
    main(output=args.output, p=args.p, sigma=args.sigma)
