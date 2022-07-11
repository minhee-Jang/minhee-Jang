import torch.nn as nn

class RecursiveFilter(nn.Module):
    def __init__(self, w=0.8):
        super().__init__()
        self.w  = w
        # self.n_frames = opt.n_frames
    # end

    def forward(self, x):
        out = x.clone().detach()
        bs, c, n, h, w = x.shape

        out[:, 0] = x[:, 0]
        # prev_x = out

        for i in range(1, n):
            out[:, :, i] = self.w * x[:, :, i] + (1 - self.w) * out[:, :, i-1]


        return out
