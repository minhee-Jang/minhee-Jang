import os
import glob
import numpy as np
import shutil
import random

from skimage.external.tifffile import imsave, imread

import argparse


def select_dir(dir):
    dirs = os.listdir(dir)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]

    path_dir = os.path.join(dir, path_opt)
    print("path_dir is: {}".format(path_dir))
    
    return path_dir

def mv2group():
    # genoray_dir = r'D:\data\denoising\test-results\genoray-20200513-waveletdl-patch80-n_resblocks32-swt-haar-lv2-serial-n_feats96-cl_l1-sl_l1-perceptual_loss-srgan-patch_offset15'
    genoray_root = r'D:\data\denoising\test-results'
    genoray_dir = select_dir(genoray_root)
    path_opt = os.path.basename(genoray_dir)
    genoray_result = r'D:\data\genoray-results'

    img_list =os.listdir(genoray_dir)

    # print(img_list)
    for p in img_list:
        group = p[4:-8]
        if group[-1] == '-':
            group = group[:-1]
        print(group)

        result_dir = os.path.join(genoray_result, path_opt, group)
        os.makedirs(result_dir, exist_ok=True)

        src = os.path.join(genoray_dir, p)
        dst = os.path.join(result_dir, p)

        shutil.move(src, dst)

def copy2group():
    genoray_low_dir = '../../data/denoising/train/genoray/Low'
    genoray_avg_dir = '../../data/denoising/train/genoray/Low_avg'
    
    genoray_low = '../../data/denoising/train/genoray-results/genoray/Low'
    genoray_avg = '../../data/denoising/train/genoray-results/genoray/Low_avg'

    low_list =os.listdir(genoray_low_dir)
    avg_list = os.listdir(genoray_avg_dir)

    # print(img_list)
    for lp, ap in zip(low_list, avg_list):
        low_group = lp[:-8]
        avg_group = ap[:-8]

        assert low_group == avg_group
        
        group = low_group
        if group[-1] == '-':
            group = group[:-1]
        print(group)

        low_dir = os.path.join(genoray_low, group)
        avg_dir = os.path.join(genoray_avg, group)
        os.makedirs(low_dir, exist_ok=True)
        os.makedirs(avg_dir, exist_ok=True)

        low_src = os.path.join(genoray_low_dir, lp)
        low_dst = os.path.join(low_dir, lp)
        avg_src = os.path.join(genoray_avg_dir, ap)
        avg_dst = os.path.join(avg_dir, lp)

        print("copying {}".format(lp))
        shutil.copy(low_src, low_dst)
        shutil.copy(avg_src, avg_dst)

def avg(n_avg):
    genoray_result = r'D:\data\genoray-results'

    path_dir = select_dir(genoray_result)
    group_list = os.listdir(path_dir)

    for group in group_list:
        img_dir = os.path.join(path_dir, group)
        print("img_dir:", img_dir)
        img_list = os.listdir(img_dir)
        n_img = random.sample(img_list, n_avg)
        print(n_img)

        ref_img_path = os.path.join(img_dir, n_img[0])
        ref_img = imread(ref_img_path)

        avg_img = np.zeros(ref_img.shape, dtype=ref_img.dtype)
        print("ref_img.shape:", ref_img.shape)
        print("avg_img.shape:", avg_img.shape)

        for img_name in n_img:
            img_path = os.path.join(img_dir, img_name)
            img = imread(img_path)

            avg_img = avg_img + img

        avg_img = avg_img / n_avg

        avg_save_path = os.path.join(img_dir, 'avg' + str(n_avg) + '-' + group + '.tiff')
        # avg_save_path = os.path.join(img_dir, 'avg' + str(n_avg) + group)
        imsave(avg_save_path, avg_img)
        # os.remove(avg_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoising models')
    parser.add_argument('--mode', type=str, default='mv2group', choices=['mv2group', 'avg', 'copy2group'])
    parser.add_argument('--n_avg', type=int, default=5)

    opt = parser.parse_args()
    if opt.mode == 'mv2group':
        mv2group()
    elif opt.mode == 'avg':
        avg(opt.n_avg)
    elif opt.mode == 'copy2group':
        copy2group()
