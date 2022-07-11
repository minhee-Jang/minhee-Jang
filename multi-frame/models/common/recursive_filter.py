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
            out = reout
        return out

