from pathlib import Path
import shutil
import math
import json

from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from rasterio.transform import xy
import torchvision.transforms as transforms


class RasterDataset(Dataset):

    def __init__(self, opt, chip_size=480, stride=None):
        imgs = opt.dataroot
        self.isTest = True
        self.ext = ['.jpg', '.png', '.img', '.tif']
        imgs = Path(imgs)
        self.imgs = imgs
        transform_list = [transforms.ToPILImage(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]
        self.pytorch_transforms = transforms.Compose(transform_list)
        assert imgs.exists() and imgs.is_dir(), "No such file or dir: {}".format(imgs)
        a_path = imgs/'A'
        b_path = imgs/'B'
        assert a_path.exists() and b_path.exists(), \
            ("source: {} should contain A and B subdirs".format(imgs))
        a_images = [x.name for x in a_path.iterdir()]
        b_images = [x.name for x in b_path.iterdir()]
        a_images.sort()
        b_images.sort()
        assert a_images == b_images
        self.img_seq = a_images
        self.img_padding_seq = []
        self.img_size_seq = []
        self.chip_size = chip_size
        self.stride = stride if stride is not None else chip_size // 2
        self.clip_stride = self.chip_size - self.stride
        self.idx_map = {}
        self.transforms = {}
        for img_idx, img in enumerate(self.img_seq):
            idx_base = len(self.idx_map)
            img_a_p = (imgs/'A')/img
            with rasterio.open(img_a_p) as fp:
                self.transforms[img] = fp.transform
                h, w = fp.shape
                ah, aw = h, w
                chips_i = [_ for _ in range(math.ceil((w - self.chip_size) / self.clip_stride) + 1)]
                chips_j = [_ for _ in range(math.ceil((h - self.chip_size) / self.clip_stride) + 1)]
                chips = [(i, j) for i in chips_i for j in chips_j]
                self.img_padding_seq.append((len(chips_j) * self.chip_size,
                                             len(chips_i) * self.chip_size))
                self.img_size_seq.append((h, w))
                for idx, (i, j) in enumerate(chips):
                    self.idx_map[idx+idx_base] = (img_idx, (i*self.clip_stride, j*self.clip_stride))
            img_b_p = (imgs / 'B') / img
            with rasterio.open(img_b_p) as fp:
                bh, bw = fp.shape
                assert ah == bh and aw == bw, \
                    "img: {}, width or height mismatch".format(img)

        imgs_map = {
            'chip_size': self.chip_size,
            'stride': self.stride,
            'img_count': len(self.img_seq),
            'idx_map': self.idx_map,
            'img_seq': self.img_seq
        }
        self.imgs_map = imgs_map
        self.count = 0

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        if self.count == len(self):
            self.count = 0
            raise StopIteration()
        img_idx, (local_x, local_y) = self.idx_map[idx]
        img = self.img_seq[img_idx]
        img_a_p = (self.imgs / 'A') / img
        img_b_p = (self.imgs / 'B') / img
        with rasterio.open(img_a_p) as fp:
            bands = (1, 2, 3,) if len(fp.units) >= 3 else (1,)
            a_im = fp.read(
                bands,
                window=Window(local_x, local_y, self.chip_size, self.chip_size),
                boundless=True,
                fill_value=0,
            )
            if bands == (1,):
                a_im = np.concatenate((a_im, a_im, a_im), axis=0)
            a_im = a_im.transpose((1, 2, 0))  # 3xHxW to HxWx3
            a_im = self.pytorch_transforms(a_im)
            a_im = a_im[np.newaxis, :]

        with rasterio.open(img_b_p) as fp:
            bands = (1, 2, 3,) if len(fp.units) >= 3 else (1,)
            b_im = fp.read(
                bands,
                window=Window(local_x, local_y, self.chip_size, self.chip_size),
                boundless=True,
                fill_value=0,
            )
            if bands == (1,):
                b_im = np.concatenate((b_im, b_im, b_im), axis=0)
            b_im = b_im.transpose((1, 2, 0))  # 3xHxW to HxWx3
            b_im = self.pytorch_transforms(b_im)
            b_im = b_im[np.newaxis, :]
        self.count += 1
        return {'A': a_im,
                'B': b_im,
                'A_paths': [],
                'img': img,
                'img_size': self.img_size_seq[img_idx],
                'chip_size': self.chip_size,
                'padding': self.img_padding_seq[img_idx],
                'yx': (local_y, local_x)}


if __name__ == '__main__':
    imgs = '/home/char1iez/Desktop/deeplearning/tmp/fastDet/objdetbbox/data/images'
    class Opt:
        def __init__(self, imgs):
            self.dataroot = imgs
    opt = Opt(imgs)
    dataset = RasterDataset(opt)
    print(len(dataset))
    for data in dataset:
        print(data['img'])
        print(data['yx'])
