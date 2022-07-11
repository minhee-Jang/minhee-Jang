import torch.nn as nn
import numpy as np
import torch

class EstimationDelta(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, rf, mf, move_thr, n_frames):
        _, _, _, h, w = x.shape
    
        # 프레임 한 장씩 가져와 array로 변환
        for i in range(0, n_frames):
            arr = x[:, :, i]
            arr = arr.reshape([-1, w])
            arr = np.array(arr)
            arr *= 255
            arr = arr.astype(int)
            globals()['o{}'.format(i)] = arr

        rf = rf.reshape([-1, w])
        mf = mf.reshape([-1, w])
        rf = np.array(rf)
        mf = np.array(mf)
        rf *= 255
        mf *= 255
        rf = rf.astype(int)
        mf = mf.astype(int)

        out = torch.zeros(x[:, :, 0].shape)
        dout = torch.zeros(x[:, :, 0].shape)
        dout = dout.reshape([-1, w])
        dout = np.array(dout)
        cout = dout.copy()
        tout = dout.copy()
        
        # Calculate delta
        for i in range(0, h):
            for j in range(0, w):
                intensityMean = 0
                total = 0
                for k in range(n_frames):
                    intensityMean += globals()['o{}'.format(k)][i][j]
                    if k == 0:  continue
                    subResult = globals()['o{}'.format(k)][i][j] - globals()['o{}'.format(k-1)][i][j]
                    total += abs(subResult)
                intensityMean = (intensityMean) / n_frames
                delta = total*1000 / (intensityMean**2)
                dout[i][j] = delta
        
        # Apply Gaussian filter to dout(delta result)
        import cv2
        darray = np.asarray(dout)
        darray = np.float32(darray)
        kernel1d = cv2.getGaussianKernel(5,3)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())
        dout = cv2.filter2D(darray, -1, kernel2d)
    
        # Normalize
        min_value = np.min(dout)
        max_value = np.max(dout)
        dout = (dout - min_value) * ((255-0)/(max_value - min_value)) + 0
    
        # Combine DL and RF
        for i in range(0, h):
            for j in range(0, w):
                if dout[i][j] >= move_thr:
                    cout[i][j] = mf[i][j]
                    tout[i][j] = 255
                else:
                    cout[i][j] = rf[i][j]
                    tout[i][j] = 0

        # Return output imgs
        mf = torch.tensor(mf)
        mf = mf.reshape(out.shape)
        rf = torch.tensor(rf)
        rf = rf.reshape(out.shape)
        cout = torch.tensor(cout)
        cout = cout.reshape(out.shape)
        dout = torch.tensor(dout)
        dout = dout.reshape(out.shape)
        tout = torch.tensor(tout)
        tout = tout.reshape(out.shape)

        return mf, rf, cout, dout, tout