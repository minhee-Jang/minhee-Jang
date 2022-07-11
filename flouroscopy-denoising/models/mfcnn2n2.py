import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel


class MFCNN2N2(BaseModel):
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

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.model
        model_opt = model_opt + "-n_inputs" + str(opt.n_inputs)
        model_opt = model_opt + "-ms_channels" + str(opt.ms_channels)
        model_opt = model_opt + "-growth_rate" + str(opt.growth_rate)
        model_opt = model_opt + "-n_denselayers" + str(opt.n_denselayers)
        # model_opt = model_opt + "-n_denseblocks" + str(opt.n_denseblocks)
        if opt.perceptual_loss is not None:
            model_opt = model_opt + '-perceptual_loss' + '-' + opt.perceptual_loss

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        savedir = os.path.join(opt.checkpoints_dir, model_opt)
        return savedir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        if self.perceptual_loss and self.is_train:
            self.loss_name = ['content_loss', 'style_loss']
        else:
            self.loss_name = ['content_loss']

        self.model_names = ['qenet']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.qenet = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()

            if self.perceptual_loss:
                self.perceptual_loss_criterion = PerceptualLoss(opt)

            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(self.qenet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()
        self.n_inputs = opt.n_inputs

        # url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None

    def set_input(self, input):
        x = input['x'].to(self.device)
        # print('inpu.t.shape:', x.shape)
        
        self.ix = x
        self.nx = x[:, self.n_inputs//2:self.n_inputs//2+1]
        # print('self.ix.shape:', self.ix.shape)
        # print('self.nx.shape:', self.nx.shape)
        if input['target'] is not None:
            self.target = input['target'].to(self.device)
            self.ct = self.target[:, self.n_inputs//2:self.n_inputs//2+1]

    def forward(self):
        self.out = self.qenet(self.ix)

    def backward(self):
        if self.perceptual_loss:
            self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.nx, self.out)
            self.loss = self.content_loss + self.style_loss
        else:
            self.loss = self.content_loss_criterion(self.nx, self.out)

        self.loss.backward()

        mse_loss = self.mse_loss_criterion(self.out.detach(), self.ct.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        self.backward()
        self.optimizerQ.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        if self.perceptual_loss:
            print("Content Loss: {:.8f}, Style Loss: {:.8f}".format(
                self.content_loss, self.style_loss)
            )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )



def create_model(opt):
    return QENet(opt)

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=3//2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=5//2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=7//2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        out = torch.cat((x3, x5, x7), dim=1)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        # print('x.shape:', x.shape)
        # print('self.layers(x).shape:', self.layers(x).shape)
        # print('self.lff(self.layers(x)).shape', self.lff(self.layers(x)).shape)
        # return x + self.lff(self.layers(x))  # local residual learning
        return self.lff(self.layers(x))


class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
    # def __init__(self, opt):
        super(RDN, self).__init__()

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        global_res = x
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x) + global_res
        return x


class QENet(nn.Module):
    def __init__(self, opt):
        super(QENet, self).__init__()
        self.n_inputs = opt.n_inputs
        n_channels = opt.n_channels
        self.nc = n_channels
        ms_channels = opt.ms_channels # number of channels of output multi-scale conv

        dense_in_channels = n_channels * ms_channels * 3 * self.n_inputs

        n_denselayers = opt.n_denselayers
        # n_denseblocks = opt.n_denseblocks # Righit now, we use one dense block
        growth_rate = opt.growth_rate

        multiscale_conv = [MultiScaleConv(n_channels, ms_channels) for _ in range(self.n_inputs)]

        self.multiscale_conv = nn.ModuleList(multiscale_conv)
        self.rdn = RDB(dense_in_channels, growth_rate, n_denselayers)
        self.tail = nn.Conv2d(growth_rate, n_channels, 3, padding=3//2)
        
    def forward(self, x):
        # print('x.shape:', x.shape)
        assert (x.size(1) // self.nc) == self.n_inputs

        # we will take mean of x-ray images
        x_mean = torch.zeros(x[:,:self.nc].shape, dtype=x.dtype, device=x.device)
        for i in range(self.n_inputs):
            x_mean = x_mean + x[:, i * self.nc: i * self.nc + self.nc]
        x_mean = x_mean / self.n_inputs
        # print('x_mean.shape:', x_mean.shape)
        
        # multi-scale network
        ms_out = []
        for i in range(self.n_inputs):
            x_in  = x[:, i*self.nc:i*self.nc+self.nc]
            ms_conv = self.multiscale_conv[i]
            ms_out.append(ms_conv(x_in))

        # densely connected network
        ms_out = torch.cat(ms_out, dim=1)
        out = self.rdn(ms_out)
        out = self.tail(out) + x_mean

        return out
