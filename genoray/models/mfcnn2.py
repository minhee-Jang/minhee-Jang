import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel
from .mfcnn import create_model as create_qenet

import numpy as np

class MFCNN2(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Network parameters
        # n_inputs is set in base options
        # n_channels is set in dataset options
        parser.add_argument('--ms_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        parser.add_argument('--growth_rate', type=int, default=32,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_denselayers', type=int, default=5,
            help='number of layers in dense block')
        # n_denseblocks is currently is not used
        # parser.add_argument('--n_denseblocks', type=int, default=8,
        #     help='number of layers of dense blocks')

        # n_denseblocks = opt.n_denseblocks # Righit now, we use one dense block

        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')

        if is_train:
            parser = parse_perceptual_loss(parser)
        else:
            parser.set_defaults(test_patches=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        self.model_names = ['qenet']
        self.n_frames = opt.n_frames

        # Create model
        self.qenet = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            if self.perceptual_loss:
                self.loss_criterion = PerceptualLoss(opt)

            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(self.qenet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        # assert len(input) == 1

        self.x = input['lr'].to(self.device)
        # b, c, n, h, w = self.x.shape
        # self.x = self.x.view(b, c*n, h, w)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)[:, :, self.n_frames//2]

    def forward(self):
        self.out = self.qenet(self.x)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizerQ.step()

    def calc_loss(self):
        if self.perceptual_loss:
            self.content_loss, self.style_loss = self.loss_criterion(self.target, self.out)
            self.loss = self.content_loss + self.style_loss
        else:
            self.loss = self.loss_criterion(self.target, self.out)
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def get_logs(self):
        if self.perceptual_loss:
            log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            'content_loss': '{:.8f}'.format(self.content_loss),
            'style_loss': '{:.8f}'.format(self.style_loss),
            # 'mse_loss': '{:.8f}'.format(mse_loss),
            'psnr': '{:.8f}'.format(self.psnr)
        }
        else:
            log_dict = {
                'loss': '{:.8f}'.format(self.loss),
                'psnr': '{:.8f}'.format(self.psnr)
            }
        return log_dict

    def get_batch_measure(self):
        return self.loss.detach(), self.psnr.detach()
        

    def predict(self, batch):
        n_frames = self.n_frames
        x = batch['lr']
        _, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []

        for d in range(n):
            predicted_idx = d + 1
            # print('d: {}, predicted_idx: {}'.format(d, predicted_idx))
            if predicted_idx % 10 != 0:
                continue

            if d < n_frames//2:
                n_dummy_vids = n_frames//2 - d
                dummy_x =  x.repeat(1, 1, n_dummy_vids, 1, 1)
                xd = x[:, :, :d + n_frames//2 + 1]
                xd = torch.cat((dummy_x, xd), dim=2)
            elif d >= n - n_frames//2:
                n_dummy_vids = n_frames//2 - (n - d) + 1
                xd = x[:, :, d - n_frames//2:]
                dummy_x =  x.repeat(1, 1, n_dummy_vids, 1, 1)
                xd = torch.cat((xd, dummy_x), dim=2)
            else:
                xd = x[:, :, d-n_frames//2:d+n_frames//2+1]
            
            tensors_input = {
                "lr": xd,
            }

            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()

            out = self.out.detach()
            out = out.permute(0, 2, 3, 1).squeeze().to('cpu').numpy()

            out[out > 1.0] = 1.0
            out[out < 0.0] = 0.
            out = out * 256
            out = np.rint(out)
            out  = out.astype(np.uint8)

            print('predicted file {:03d}'.format(predicted_idx))
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        return predicted_video, predicted_idxs

