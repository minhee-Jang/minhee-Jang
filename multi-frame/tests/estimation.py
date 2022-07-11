import torch.nn as nn
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class EstimationSTD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_frames = opt.n_frames
    # end
    
    def forward(self, x):
        out = x.clone().detach()
        bs, c, n, h, w = x.shape

        out[:, 0] = x[:, 0]
        print("out shape : ", out.shape)

        xiout = out[:, :, 0]
        print(xiout.shape)
        
        xi2 = out[:, :, 0]
        xi1 = out[:, :, 1]
        xi0 = out[:, :, 2]
        sout = xiout.reshape([-1, w])
        xi2 = xi2.reshape([-1, w])
        xi1 = xi1.reshape([-1, w])
        xi0 = xi0.reshape([-1, w])
        sout = sout.tolist()
        x2 = xi2.tolist()
        x1 = xi1.tolist()
        x0 = xi0.tolist()
        sout = np.array(sout)
        x2 = np.array(x2)
        x1 = np.array(x1)
        x0 = np.array(x0)
        print("리스트 사이즈: ", x0.shape)

        # # 완전 STD 방식
        # for i in range(0, h):
        #     for j in range(0, w):
        #         intensityList = []
        #         intensityList.append(x2[i][j])
        #         intensityList.append(x1[i][j])
        #         intensityList.append(x0[i][j])
        #         std_out = np.std(intensityList)
        #         sout[i][j] = std_out
        
        # delta 중첩 방식
        for i in range(0, h):
            for j in range(0, w):
                d1 = x1[i][j] - x2[i][j]
                d2 = x0[i][j] - x1[i][j]
                delta = d1 + d2
                sout[i][j] = delta

        min_max_scaler = MinMaxScaler()
        scaled_out = min_max_scaler.fit_transform(sout)

        sout = torch.tensor(scaled_out)
        sout = sout.reshape(xiout.shape)

        return sout
