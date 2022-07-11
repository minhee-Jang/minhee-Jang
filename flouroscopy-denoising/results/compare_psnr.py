
import os
import glob
import numpy as np
import shutil
import random
import cv2

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

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def dl_compare():
    result_dir = r'D:\data\genoray-results'
    avg_dir = r'D:\data\genoray-results\genoray\Low_avg'
    
    # psnr_log = r'D:\data\genoray-results\genoray\src_psnr_log.csv'
    dl_dir =select_dir(result_dir)
    psnr_log = os.path.join(dl_dir, 'psnr_log.csv')

    with open(psnr_log, 'w') as f:
        f.write('group,num_avg,psnr\n')

    group_list = os.listdir(dl_dir)

    for gr in group_list:
        if gr == 'psnr_log.csv': continue
        print('group', gr)
        low_img_list = os.listdir(os.path.join(dl_dir, gr))
        avg_gr_dir = os.path.join(avg_dir, gr)
        
        avg_fn = os.listdir(avg_gr_dir)[0]
        avg_img = imread(os.path.join(avg_gr_dir, avg_fn))

        sum_img = np.zeros(avg_img.shape, dtype=avg_img.dtype)
        for n, low_fn in enumerate(low_img_list, 1):
            if low_fn[:3] == 'avg':
                continue
            else:
                print('current file name:', low_fn)
            low_img = imread(os.path.join(dl_dir, gr, low_fn))

            sum_img = sum_img + low_img
            low_avg_img = sum_img / n
            # if n == 1:
            #     low_avg_img = low_img
            # else:
            #     low_avg_img = opt.recursive_weight * low_avg_img + (1 - opt.recursive_weight) * low_img

            psnr = output_psnr_mse(avg_img, low_avg_img)
            print('{} {}: {:.8f}'.format(gr, n, psnr))

            with open(psnr_log, 'a') as f:
                f.write('{},{},{:.8f}\n'.format(gr, n, psnr))


def src_compare(opt):
    low_dir = r'D:\data\genoray-results\genoray\Low'
    avg_dir = r'D:\data\genoray-results\genoray\Low_avg'
    
    psnr_log = r'D:\data\genoray-results\genoray\src_psnr_log.csv'
    group_list =os.listdir(low_dir)

    with open(psnr_log, 'w') as f:
        f.write('group,num_avg,psnr\n')

    for gr in group_list:
        print('group', gr)
        low_img_list = os.listdir(os.path.join(low_dir, gr))
        avg_gr_dir = os.path.join(avg_dir, gr)
        
        avg_fn = os.listdir(avg_gr_dir)[0]
        avg_img = imread(os.path.join(avg_gr_dir, avg_fn))

        sum_img = np.zeros(avg_img.shape, dtype=avg_img.dtype)
        for n, low_fn in enumerate(low_img_list, 1):
            low_img = imread(os.path.join(low_dir, gr, low_fn))

            sum_img = sum_img + low_img
            low_avg_img = sum_img / n
            # if n == 1:
            #     low_avg_img = low_img
            # else:
            #     low_avg_img = opt.recursive_weight * low_avg_img + (1 - opt.recursive_weight) * low_img

            psnr = output_psnr_mse(avg_img, low_avg_img)
            print('{} {}: {:.8f}'.format(gr, n, psnr))

            with open(psnr_log, 'a') as f:
                f.write('{},{},{:.8f}\n'.format(gr, n, psnr))

def compare_all(opt):
    result_dir = r'D:\data\genoray-results'
    low_dir = r'D:\data\genoray-results\genoray\Low'
    avg_dir = r'D:\data\genoray-results\genoray\Low_avg'
    
    # psnr_log = r'D:\data\genoray-results\genoray\src_psnr_log.csv'
    dl_dir =select_dir(result_dir)
    psnr_log = os.path.join(dl_dir, 'psnr_log.csv')

    with open(psnr_log, 'w') as f:
        f.write('group,num_avg,psnr\n')

    group_list = os.listdir(dl_dir)

    for gr in group_list:
        if gr == 'psnr_log.csv': continue
        print('group', gr)
        dl_img_list = os.listdir(os.path.join(dl_dir, gr))
        low_img_list = os.listdir(os.path.join(low_dir, gr))
        avg_img_list = os.listdir(os.path.join(avg_dir, gr))

        print('len(low_img_list):', len(low_img_list))
        print('len(dl_img_list):', len(dl_img_list))

        
        avg_fn = avg_img_list[0]
        avg_img = imread(os.path.join(avg_dir, gr, avg_fn))
        print('average img:", avg_fn')
        sum_img = np.zeros(avg_img.shape, dtype=avg_img.dtype)

        n = 1
        for low_fn, dl_fn in zip(low_img_list, dl_img_list):
            if dl_fn[:3] == 'avg':
                continue
            else:
                print('current file name:', dl_fn)
            
            low_img = imread(os.path.join(low_dir, gr, low_fn))
            dl_img = imread(os.path.join(dl_dir, gr, dl_fn))

            sum_img = sum_img + dl_img
            dl_avg_img = sum_img / n
            
            # if n == 1:
            #     low_avg_img = low_img
            # else:
            #     low_avg_img = opt.recursive_weight * low_avg_img + (1 - opt.recursive_weight) * low_img

            # print('n:', n)
            # print('low_img')
            # print_numpy(low_img)
            # print('dl_img')
            # print_numpy(dl_img)
            # print('dl_avg_img')
            # print_numpy(dl_avg_img)
            # print('avg_img')
            # print_numpy(avg_img)
            
            # compare_img = np.concatenate((low_img, dl_img, dl_avg_img), axis=1)

            # cv2.imshow("compare_img", compare_img)
            # cv2.waitKey()

            psnr = output_psnr_mse(avg_img, dl_avg_img)
            print('{} {}: {:.8f}'.format(gr, n, psnr))

            with open(psnr_log, 'a') as f:
                f.write('{},{},{:.8f}\n'.format(gr, n, psnr))

            n += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoising models')
    parser.add_argument('--mode', type=str, default='src_compare', choices=['dl_compare', 'src_compare', 'compare_all'])
    parser.add_argument('--recursive_weight', type=float, default=0.9)
    parser.add_argument('--n_avg', type=int, default=5)
    opt = parser.parse_args()

    if opt.mode == 'dl_compare':
        dl_compare()
    elif opt.mode == 'src_compare':
        src_compare(opt)
    elif opt.mode == 'compare_all':
        compare_all(opt)


