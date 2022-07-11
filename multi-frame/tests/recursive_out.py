import os, sys
utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)
from data import create_dataset
from options.test_options import TestOptions
from skimage.io import imsave
import torch.nn as nn

class RecursiveFilter(nn.Module):
    def __init__(self, w=0.2):
        super().__init__()
        self.w  = w

    def forward(self, x):
        reout = x.clone().detach()
        _, _, n, _, _ = x.shape

        reout[:, :, 0] = x[:, :, 0]

        for i in range(1, n):
            reout[:, :, i] = self.w * x[:, :, i] + (1 - self.w) * reout[:, :, i-1]

        return reout

if __name__ == '__main__':
    opt = TestOptions(r'D:\\data\\video-sr').parse()
    opt.n_frames = 5
    dataloader = create_dataset(opt)
    n_frames = opt.n_frames

    recursive_filter = RecursiveFilter(w=0.2)

    test_dir = opt.test_results_dir + '-rf/0623_0.2_new_5frames/'
    os.makedirs(test_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xAll = batch['lr']    #low
        bs, c, n, h, w = xAll.shape

        rfAll = recursive_filter(xAll)
        rfAll = rfAll.to('cuda')

        _, _, r_frames, _, _ = rfAll.shape

        # 000~009까지는 10 frames 다 쌓이지 않았으므로 기존 방식으로 쌓음
        for j in range(n_frames):
            print('j: ', j)
            rfIdx = '{:03d}'.format(j)
            
            rf = rfAll[:, :, j]
            rout = rf.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_path = os.path.join(test_dir, rfIdx + '.tiff')
            imsave(recursive_path, rout)


        # 010~191까지는 10 frames씩 가져와서 쌓음
        for k in range(n_frames, r_frames):
            print('k:', k)
            rfIdx = '{:03d}'.format(k)
            
            x = xAll[:, :, k-n_frames:k+1]   # 000~191 (rf쌓을때000부터필요)
            rfAll2 = recursive_filter(x)
            
            rf = rfAll2[:, :, n_frames]
            rout = rf.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_path = os.path.join(test_dir, rfIdx + '.tiff')
            imsave(recursive_path, rout)