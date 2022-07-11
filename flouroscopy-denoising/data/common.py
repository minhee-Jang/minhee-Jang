
import random
import numpy as np
import skimage.color as sc

import torch

def add_noise(x, noise=0):
    if noise == 0:
        noise_value = np.random.randint(55)
    else:
        noise_value = noise

    noises = np.random.normal(scale=noise_value, size=x.shape)
    noises = noises.round()
        
    x_noise = x.astype(np.int16) + noises.astype(np.int16)
    x_noise = x_noise.clip(0, 255).astype(np.uint8)
    return x_noise

def augment(*args, hflip=True, rot=True):
    # print('len(args):', len(args))
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # print('type(img)', type(img))
        if img.ndim == 2:
            if hflip: img = img[:, ::-1].copy()
            if vflip: img = img[::-1, :].copy()
            if rot90: img = img.transpose(1, 0).copy()
        elif img.ndim == 3:
            if hflip: img = img[:, ::-1, :].copy()
            if vflip: img = img[::-1, :, :].copy()
            if rot90: img = img.transpose(1, 0, 2).copy()
            
        return img

    return [_augment(a) for a in args]

def get_patch(*args, patch_size=96, n_channels=1, scale=1, multi=False, input_large=False):
    lr_n = args[0]
    # print('len(lr_n):', len(lr_n))
    # print('len(hr)', len(hr))
    # print('type(hr):', type(hr))
    ih, iw = lr_n[0].shape[:2]
    # print('ih: {}, iw: {}'.format(ih, iw))
    # ih, iw = args[0].shape[:2]

    # tp = patch_size
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    # tx, ty = ix, iy

    # if n_channels == 1:
    #     ret = [
    #         args[0][iy:iy + ip, ix:ix + ip],
    #         *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
    #     ]
    # else:
    #     ret = [
    #         args[0][iy:iy + ip, ix:ix + ip, :],
    #         *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    #     ]

    if n_channels == 1:
        ret = [
            *[a[iy:iy + ip, ix:ix + ip] for a in args[0]],
            *[a[iy:iy + ip, ix:ix + ip] for a in args[1]]
        ]
    else:
        ret = [
            *[a[iy:iy + ip, ix:ix + ip, :] for a in args[0]],
            *[a[iy:iy + ip, ix:ix + ip, :] for a in args[1]],
        ]

    # print('len(ret[0]):', len(ret[0]))
    # print('len(ret[1]):', len(ret[1]))
    # lr_n = ret[:-1]
    # hr = ret[-1]
    # print('len(ret):', len(ret))
    # print('type(hr):', type(hr))
    # print('hr.shape:', hr.shape)
    # for p in ret:
    #     print('type(p):', type(p))
    #     print('p.shape:', p.shape)
    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / rgb_range)

        return tensor

    return [_np2Tensor(a) for a in args]
