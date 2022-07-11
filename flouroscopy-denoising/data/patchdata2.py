import os
import glob
import pickle
import numpy as np
import torch

# from skimage import io
# from skimage.external import tifffile
import imageio
import random

from . import common
from .base_dataset import BaseDataset

class PatchData2(BaseDataset):
    def __init__(self, args, name='', is_train=True):
        self.args = args
        self.dataset = name
        self.in_mem = args.in_mem
        self.n_inputs = args.n_inputs
        self.n_channels = args.n_channels

        self.is_train = is_train
        # self.benchmark = benchmark
        self.test_random_patch = args.test_random_patch
        self.mode = 'train' if is_train else 'test'

        self.add_noise = args.add_noise
        self.noise = 0
        
        self._set_filesystem(args.data_dir)
        print("----------------- {} {} dataset -------------------".format(name, self.mode))
        print("Set file system for {} dataset {}".format(self.mode, self.dataset))
        print("apath:", os.path.abspath(self.apath))
        print("dir_hr:", os.path.abspath(self.dir_hr))
        print("dir_lr:", os.path.abspath(self.dir_lr))
        print("----------------- End ---------------------------")

        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        _, _, self.case_list, self.hr_dict, self.lr_dict = self._scan()

        self.images_hr = {}
        self.images_lr = {}
        for case in self.case_list:
            if args.ext.find('img') >= 0:
                print("{} image loading".format(__file__))
                self.images_hr[case], self.images_lr[case] = self.hr_dict[case], self.lr_dict[case]
            elif args.ext.find('sep') >= 0:
                self.images_hr[case], self.images_lr[case] = [], []
                for h in self.hr_dict[case]:
                    b = h.replace(self.apath, path_bin)
                    os.makedirs(os.path.dirname(b), exist_ok=True)

                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr[case].append(b)
                    self._check_and_load(args.ext, h, b, verbose=True) 
                for l in self.lr_dict[case]:
                    b = l.replace(self.apath, path_bin)
                    os.makedirs(os.path.dirname(b), exist_ok=True)

                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[case].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)

        if self.in_mem:
            self._load2mem()
        
        if self.is_train:
            n_patches = args.batch_size * args.test_every
            # n_images = len(args.datasets) * len(self.images_hr)
            n_images = len(args.datasets) * len(self.case_list)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.mode, self.dataset)
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                # if self.args.n_channels == 1:
                #     pickle.dump(t_imread(img), _f)
                # else : 
                #     pickle.dump(imread(img), _f)
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if not self.in_mem:
            # lr_n, hr, filename, case = self._load_file(idx)
            lr_n, hr_n, case, cidx = self._load_file(idx)
        else:
            # lr_n, hr, filename, case = self._load_mem(idx)
            lr_n, hr_n, case, cidx = self._load_mem(idx)

        if self.is_train:
            pair = self.get_patch(lr_n, hr_n)
        else:
            # pair = [lr_n, hr]
            # print('len(lr_n):', len(lr_n))
            # print('len(hr_n):', len(hr_n))
            lr_n.extend(hr_n)
            pair = lr_n
            # print('len(pair):', len(pair))
            

        pair = common.set_channel(*pair, n_channels=self.n_channels)
        # if self.n_channels == 3: pair = [(p.astype(np.float) / 255.0) for p in pair]
        # pair = lr_n.expand(hr_n)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        # print('len(pair_t):', len(pair_t))
        # print('pair_t.shape:', pair_t.shape)
        n = len(pair_t)

        lr_n = torch.cat(pair_t[:n//2], dim=0)
        hr_n = torch.cat(pair_t[n//2:], dim=0)
        # hr = pair_t[-1]
        # print('lr_n.shape:', lr_n.shape)
        # print('hr_n.shape:', hr_n.shape)
        # return lr_n, hr, filename, case
        return lr_n, hr_n, case, cidx
            
    def __len__(self):
        if self.is_train:
            # return len(self.images_hr) * self.repeat
            return len(self.case_list) * self.repeat
        else:
            # return len(self.images_hr)
            return len(self.case_list)

    def _get_index(self, idx):
        if self.is_train:
            # return idx % len(self.images_hr)
            return idx % len(self.case_list)
        else:
            return idx

    def get_sequences(self, case):
        total_n = len(self.images_lr[case])
        idx = 0
        if self.is_train:
            r = self.n_inputs // 2
            idx = random.randint(r, total_n - r - 1)

            # print('total_n: {}, idx: {}'.format(total_n, idx))
            idx_min = idx - r
            idx_max = idx + r
            lr, hr = self.images_lr[case][idx_min:idx_max+1], self.images_hr[case][idx_min:idx_max+1]
        else:
            lr = self.images_lr[case]
            hr = self.images_hr[case]

        return lr, hr, idx
        
    def _load_file(self, idx):
        idx = self._get_index(idx)
        case = self.case_list[idx]
        f_lr_n, f_hr_n, cidx = self.get_sequences(case)

        # filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img':
            hr_n = [imageio.imread(f_hr) for f_hr in f_hr_n]
            lr_n = [imageio.imread(f_lr) for f_lr in f_lr_n]

        elif self.args.ext.find('sep') >= 0:
            hr_n, lr_n = [], []
            for f_hr, f_lr in zip(f_hr_n,f_lr_n):
                with open(f_hr, 'rb') as _f:
                    hr_n.append(pickle.load(_f))
                with open(f_lr, 'rb') as _f:
                    lr_n.append(pickle.load(_f))
                
        hr_n = [np.asarray(hr) for hr in hr_n]
        lr_n = [np.asarray(lr) for lr in lr_n]
            
        # return lr_n, hr, filename, case
        return lr_n, hr_n, case, cidx

    def _load_mem(self, idx):
        idx = self._get_index(idx)
        # lr = self.images_lr[idx]
        # hr = self.images_hr[idx]
        case = self.case_list[idx]
        lr_n, hr_n = self.get_sequences(case)
        # filename = self.filename_list[case]

        # print('len(lr_n):', len(lr_n))
        # for lr in lr_n:
        #     print('type(lr):', type(lr))
        # print('len(hr):' , len(hr))
        # print('filename', filename)

        # return lr_n, hr, filename, case
        return lr_n, hr_n, case, cidx

    def _load2mem(self):
        # for key in self.images_hr:
        #     print("key: {}".format(key))
        images_hr_list = {}
        images_lr_list = {}
        self.filename_list = {}
        for case in self.case_list:
            # print(case)
            images_hr_list[case] = []
            images_lr_list[case] = []

            f_hr = self.images_hr[case][0]
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
                hr = np.asarray(hr)
                images_hr_list[case].append(hr)

            f_lr_all = self.images_lr[case]
            for f_lr in f_lr_all:
                with open(f_lr, 'rb') as _f:
                    lr = pickle.load(_f)
                    lr = np.asarray(lr)
                    images_lr_list[case].append(lr)

            filename, _ = os.path.splitext(os.path.basename(f_hr))
            self.filename_list[case] = filename

        # print('ended')
        self.images_hr = images_hr_list
        self.images_lr = images_lr_list

    def get_patch(self, lr, hr):
        pair = common.get_patch(
            lr, hr,
            patch_size=self.args.patch_size,
            n_channels=self.n_channels
        )
        if self.args.augment: pair = common.augment(*pair)
        # if self.add_noise: lr = common.add_noise(lr, self.noise)

        # return lr, hr
        return pair
