import os, sys

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)

from data import create_dataset
from options.train_options import TrainOptions
# from models import create_model
# from utils.utils import print_numpy




if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataloader = create_dataset(opt)

    for i, batch in enumerate(dataloader['test']):
        x, target, case, cidx = batch
        bs, nc, h, w = x.shape

        print("x.shape:", x.shape)
        print("target.shape:", target.shape)

        for b in range(bs):
            # print("batch:", b)
            print('{}-{}'.format(case[b], cidx[b]))
            x_img = x[b].to('cpu').data.numpy()
            target_img = target[b].to('cpu').data.numpy()

            x_img = x_img.transpose((1, 2, 0))
            target_img = target_img.transpose((1, 2, 0))

            # print('x_img.shape:', x_img.shape)
            concat_x_img = []
            concat_t_img = []
            for c in range(nc):
                # print("x_img[:, :, {}].shape: {}".format(c, x_img[:, :, c].shape))
                concat_x_img.append(x_img[:, :, c].squeeze())
                concat_t_img.append(target_img[:, :, c].squeeze())

            # concat_img.append(target_img.squeeze())
            concat_x_img = np.concatenate(concat_x_img, axis=1)
            concat_t_img = np.concatenate(concat_t_img, axis=1)
            concat_img = np.concatenate((concat_x_img, concat_t_img), axis=0)

            plt.imshow(concat_img, cmap=plt.cm.gray)
            plt.show()
