import os, sys
import glob

import numpy as np
from skimage.transform import rescale
import torch
import matplotlib.pyplot as plt
from skimage.io import imsave

utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)

from data import create_dataset
from options.train_options import TrainOptions
from options.test_options import TestOptions

#from utils.misc import print_numpy
from models.common.spynet import SpyNet
from models import create_model
from models.common.recursive_filter import RecursiveFilter

from models.common.flow_vis import flow_to_color
from models.mmedit.models.common.flow_warp import flow_warp

def normalize_flow(flow):
    x = flow.clone()
    bs, _, h, w = flow.shape
    x = x.view(flow.size(0), -1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(bs, 1, h, w)
    return x

if __name__ == '__main__':
    move_thr = 0.5
    opt = TestOptions(r'D:\data\video-sr').parse()
    dataloader = create_dataset(opt)

    n_frames = opt.n_frames
    center = n_frames // 2
    n_frames2 = n_frames + 7
    spynet = SpyNet().to('cuda')
    mfcnn = create_model(opt)
    recursive_filter = RecursiveFilter(w=0.2)

    test_dir = opt.test_results_dir + '-recursive-flow'
    os.makedirs(test_dir, exist_ok=True)
    mfcnn_dir = os.path.join(test_dir, 'mfcnn')
    os.makedirs(mfcnn_dir, exist_ok=True)
    recursive_dir = os.path.join(test_dir, 'recursive')
    os.makedirs(recursive_dir, exist_ok=True)
    warped_dir = os.path.join(test_dir, 'wapred')
    os.makedirs(warped_dir, exist_ok=True)
    flow_dir = os.path.join(test_dir, 'flow')
    os.makedirs(flow_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xn = batch['lr']
        tn = batch['hr']
        filenames = batch['filenames']
        print("xn.shape:", xn.shape)
        # print(filenames)
       
        bs, c, n, h, w = xn.shape

        rf_out = recursive_filter(xn)
        rf_out = rf_out.to('cuda')
        print('rf_out.shape:', rf_out.shape)
        
        for i in range(n - n_frames):
            print('i:', i)
            xi = xn[:, :, i:i+n_frames]
            ti = tn[:, :, i:i+n_frames]
            rf = rf_out[:, :, i:i+1]
            fns = filenames[i:i+n_frames]
            bs, c, n, h, w = xi.shape

            print('rf.shape:', rf.shape)
  

            tensors_dict = {
                'lr': xi,
                'target': ti,
                # 'filename': batch['filename']
            }
            mfcnn.set_input(tensors_dict)
            mfcnn.test()
            mfcnn_out = mfcnn.out
            print("mfcnn")
            print(np.max(mfcnn_out), np.min(mfcnn_out))
            mfcnn_out=((mfcnn_out-np.min(mfcnn_out))*255)/(np.max(mfcnn_out)-np.min(mfcnn_out))
            print(np.max(mfcnn_out), np.min(mfcnn_out))
            fn_center = fns[center][0]
            print('fn_center:', fn_center)

            

            #rf_center = rf[:, :, n_frames2-1]
            rf = rf.squeeze(0)
            print('rf:', rf.shape)
            print("rf")
            print(np.max(rf), np.min(rf))
            rf=((rf-np.min(rf))*255)/(np.max(rf)-np.min(rf))
            print(np.max(rf), np.min(rf))
            recursive_center = rf.detach().to('cpu').numpy().transpose(0, 2, 3, 1)
           

            recursive_path = os.path.join(recursive_dir, fn_center + '.tiff')
            imsave(recursive_path, recursive_center)
            mfcnn_path = os.path.join(mfcnn_dir, fn_center + '.tiff')
            mfcnn_out_np = mfcnn_out.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            imsave(mfcnn_path, mfcnn_out_np)

           



