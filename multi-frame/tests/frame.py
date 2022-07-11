import os, sys
import glob
import cv2
from cv2 import bilateralFilter
import numpy as np
from skimage.transform import rescale
import torch
import matplotlib.pyplot as plt
from skimage.io import imsave
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

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

def estimation(x_in, t_in, mf_in, move_thr,test_dir, n_frame2,id):
        #out = x_in.clone().detach()
        rf_center = t_in[:,:, 4:]

        
        #print(out.shape)
        xiout = torch.zeros(x_in[:,:,0].shape)
        print("hre",xiout.max())
        print(xiout.shape)
        x0 = x_in[:, :, :1]
        x1 = x_in[:, :, 1:2]
        x2 = x_in[:, :, 2:3]
        x3 = x_in[:,:,3:4]
        x4 = x_in[:,:,4:5]
        mf1 = mf_in[:,:,4:5]

        sout = xiout.reshape([-1, w])
        x0 = x0.reshape([-1, w])
        x1 = x1.reshape([-1, w])
        x2 = x2.reshape([-1, w])
        x3 = x3.reshape([-1, w])
        x4 = x4.reshape([-1, w])
        mf1 = mf1.reshape([-1, w])
        rf_center = rf_center.reshape([-1,w])
        #min_max_scaler = MinMaxScaler()
        #rf_center = min_max_scaler.fit_transform(rf_center)
        print(mf1.shape)
        mf1 = np.array(mf1)
        mf1 *= 255
        mf1 = mf1.astype(int)
        print(np.max(mf1))
        print(np.min(mf1))
        # mf1=((mf1-np.min(mf1))*255)/(np.max(mf1)-np.min(mf1))
        # print("normalization")
        # print(np.max(mf1))
        # print(np.min(mf1))
        # mf1 = mf1/255
        # mf1 = mf1.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
        # mfcnn_path = os.path.join(mfcnn_dir, fn_center  + '.tiff')
        # imsave(mfcnn_path, mf1)



        rf_center = np.array(rf_center)
        rf_center *= 255
        rf_center = rf_center.astype(int)
        # print("rf")
        # rf_center=((rf_center-np.min(rf_center))*255)/(np.max(rf_center)-np.min(rf_center))
        # print(np.max(rf_center))
        # print(np.min(rf_center))


        sout = np.array(sout)
        sout *= 255
        sout = sout.astype(int)
  
        print("sout")
        print(np.max(sout))
        print(np.min(sout))

        x2 = np.array(x2)
        x2 *= 255
        x2 = x2.astype(int)
       # print(np.max(x2))

        x1 = np.array(x1)
        x1 *= 255
        x1 = x1.astype(int)
        #print(np.max(x1))

        x0 = np.array(x0)
        x0 *= 255
        x0 = x0.astype(int)
       # print(np.max(x0))

        x3 = np.array(x3)
        x3 *= 255
        x3 = x3.astype(int)
       # print(np.max(x3))

        x4 = np.array(x4)
        x4 *= 255
        x4 = x4.astype(int)
        #print(np.max(x4))




        for i in range(0, h):
            for j in range(0, w):

                d1 = x2[i,j] - x1[i,j]
                d2 = x1[i,j] - x0[i,j]
                d3 = x3[i,j] - x2[i,j]
                d4 = x4[i,j] - x3[i,j]
                mean = (x0[i,j]+ x1[i,j ] +x2[i,j] +x3[i,j] +x4[i,j])/5
                delta = (abs(d1) + abs(d2) +abs(d3) +abs(d4)) *50 / mean
                #delta = abs(d1) + abs(d2) +abs(d3) +abs(d4)
                sout[i,j] = delta 
            #print(x2[0][0])

        moving_map = sout

        moving_map=moving_map.astype(np.uint8)
        print(moving_map.dtype)
        #print("sout max=", np.max(sout))
        moving_map= bilateralFilter(moving_map, 1, 50, 50)

        
        moving_map = torch.tensor(moving_map)
        moving_map = moving_map.reshape(xiout.shape) 
        
        

        print("************************")
        print(np.max(sout))
        print(np.min(sout))
        for y in range(0, h):
            for x in range(0, w):
                if sout[y,x]<= move_thr3 : 
                   sout[y,x] = rf_center[y,x]
                else:
                    sout[y,x] = mf1[y,x]
        
        #sout=sout.astype(np.uint8)
        #print(sout.dtype)
        print("sout max=", np.max(sout))
        #sout= bilateralFilter(sout, 1, 50, 50)
        sout = torch.tensor(sout)
        sout = sout.reshape(xiout.shape)
        
        
        #sout = torch.tensor(sout, dtyp8e=torch.int32)
        return sout, moving_map

    

if __name__ == '__main__':
    move_thr3 = 25
    opt = TestOptions(r'D:\\data\\video-sr').parse()
    dataloader = create_dataset(opt)

    n_frames = opt.n_frames
    n_frame2 = n_frames + 7
    center = n_frames // 2
    center2 = n_frame2 //2

    recursive_filter = RecursiveFilter(w=0.2)

    test_dir = opt.test_results_dir + '-recursive-flow'
    os.makedirs(test_dir, exist_ok=True)
    mfcnn_dir = os.path.join(test_dir, 'mfcnn')
    os.makedirs(mfcnn_dir, exist_ok=True)
    # recursive_dir = os.path.join(test_dir, 'recursive')
    # os.makedirs(recursive_dir, exist_ok=True)
    warped_dir = os.path.join(test_dir, 'genorary(mean_bilateral)25')
    os.makedirs(warped_dir, exist_ok=True)
    moving_dir = os.path.join(test_dir, 'moving_map(mean_bilateral)')
    os.makedirs(moving_dir, exist_ok=True)
    # flow_dir = os.path.join(test_dir, 'flow')
    # os.makedirs(flow_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xn = batch['lr']
        print(len(xn))
        tn = batch['hr']
        print(len(tn))
        mf = batch['mf']

        filenames = batch['filenames']
        bs, c, n, h, w = xn.shape


        for i in range(n - n_frames):
            
            id = i
            print('i:', id)
            xi = xn[:, :, i:i+n_frames]
            mf_1 = mf[:, :, i:i+n_frames]
            ti = tn[:, :, i:i+n_frames]
           
            out_img, moving_map = estimation(xi, ti, mf_1, move_thr3,test_dir, n_frame2, id)
            print("out_img.max?", out_img.max())
            
            fns = filenames[i:i+n_frames]
            bs, c, n, h, w = xi.shape

            fn_center = fns[center][0]
            #print("here")
            #print(np.max(out_img))
            out_img = out_img/255
            print("convert", out_img.max())
            out_img = out_img.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            warped_path = os.path.join(warped_dir, fn_center  + '.tiff')
            imsave(warped_path, out_img)

            moving_map = moving_map/255
           
            moving_map = moving_map.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            moving_path = os.path.join(moving_dir, fn_center  + '.tiff')
            imsave(moving_path, moving_map)


        


