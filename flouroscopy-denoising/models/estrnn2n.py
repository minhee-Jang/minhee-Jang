"""
https://github.com/zzh-tech/ESTRNN
"""

import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel

class ESTRNN(BaseModel):
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
        parser.set_defaults(patch_size=96)
        parser.add_argument('--n_features', type=int, default=16,
            help='base # of channels for Conv')
        parser.add_argument('--n_blocks', type=int, default=15,
            help='# of blocks in middle part of the model')
        parser.add_argument('--future_frames', type=int, default=2,
            help='use # of future frames')
        parser.add_argument('--past_frames', type=int, default=2,
            help='use # of past frames')
        parser.add_argument('--activation', type=str, default='gelu',
            help='activation function')
        # n_denseblocks is currently is not used
        # parser.add_argument('--n_denseblocks', type=int, default=8,
        #     help='number of layers of dense blocks')

        # n_denseblocks = opt.n_denseblocks # Righit now, we use one dense block

        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')


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
        model_opt = model_opt + "-n_features" + str(opt.n_features)
        model_opt = model_opt + "-n_blocks" + str(opt.n_blocks)
        model_opt = model_opt + "-future_frames" + str(opt.future_frames)
        model_opt = model_opt + "-past_frames" + str(opt.past_frames)
        # model_opt = model_opt + "-n_denseblocks" + str(opt.n_denseblocks)
        model_opt = model_opt + '-activation' + opt.activation

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

        self.model_names = ['net']

        # Create model
        self.net = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            # if opt.content_loss == 'l1':
            #     self.content_loss_criterion = nn.L1Loss()
            # elif opt.content_loss == 'l2':
            self.content_loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

        self.n_channels = opt.n_channels
        self.n_inputs = opt.n_inputs

        # url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        # print('x.shape:', self.x.shape)

        bs, c, h, w = self.x.shape
        self.x = self.x.view(bs, c // self.n_channels, self.n_channels, h, w)
        # print('x.shape:', self.x.shape)


        if input['target'] is not None:
            self.target = input['target'].to(self.device)
            self.target = self.target[:, self.n_inputs//2:self.n_inputs//2+1].view(bs, 1, self.n_channels, h, w)
            # print('target.shape:', self.target.shape)

    def forward(self):
        self.out = self.net(self.x)

    def backward(self):
        self.loss = self.content_loss_criterion(self.target, self.out)

        self.loss.backward()

        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()
    
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
    return ESTRNNModel(opt)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)


def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


def make_blocks(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


# class ResBlock(nn.Module):
#     """
#     Residual block
#     """

#     def __init__(self, in_chs, activation='relu', batch_norm=False):
#         super(ResBlock, self).__init__()
#         op = []
#         for i in range(2):
#             op.append(conv3x3(in_chs, in_chs))
#             if batch_norm:
#                 op.append(nn.BatchNorm2d(in_chs))
#             if i == 0:
#                 op.append(actFunc(activation))
#         self.main_branch = nn.Sequential(*op)

#     def forward(self, x):
#         out = self.main_branch(x)
#         out += x
#         return out


class DenseLayer(nn.Module):
    """
    Dense layer for residual dense block
    """

    def __init__(self, in_chs, growth_rate, activation='relu'):
        super(DenseLayer, self).__init__()
        self.conv = conv3x3(in_chs, growth_rate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class ResDenseBlock(nn.Module):
    """
    Residual Dense Block
    """

    def __init__(self, in_chs, growth_rate, num_layer, activation='relu'):
        super(ResDenseBlock, self).__init__()
        in_chs_acc = in_chs
        op = []
        for i in range(num_layer):
            op.append(DenseLayer(in_chs_acc, growth_rate, activation))
            in_chs_acc += growth_rate
        self.dense_layers = nn.Sequential(*op)
        self.conv1x1 = conv1x1(in_chs_acc, in_chs)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


# class RDNet(nn.Module):
#     """
#     Middle network of residual dense blocks
#     """

#     def __init__(self, in_chs, growth_rate, num_layer, num_blocks, activation='relu'):
#         super(RDNet, self).__init__()
#         self.num_blocks = num_blocks
#         self.RDBs = nn.ModuleList()
#         for i in range(num_blocks):
#             self.RDBs.append(ResDenseBlock(in_chs, growth_rate, num_layer, activation))
#         self.conv1x1 = conv1x1(num_blocks * in_chs, in_chs)
#         self.conv3x3 = conv3x3(in_chs, in_chs)
#         self.act = actFunc(activation)

#     def forward(self, x):
#         out = []
#         h = x
#         for i in range(self.num_blocks):
#             h = self.RDBs[i](h)
#             out.append(h)
#         out = torch.cat(out, dim=1)
#         out = self.act(self.conv1x1(out))
#         out = self.act(self.conv3x3(out))
#         return out


# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out



# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for _ in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out

class RDNet(nn.Module):
    """
    Middle network of residual dense blocks
    """

    def __init__(self, in_chs, growth_rate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(ResDenseBlock(in_chs, growth_rate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_chs, in_chs)
        self.conv3x3 = conv3x3(in_chs, in_chs)
        self.act = actFunc(activation)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.act(self.conv1x1(out))
        out = self.act(self.conv3x3(out))
        return out



# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# Global spatio-temporal attention module
class GSA(nn.Module):
    def __init__(self, opt):
        super(GSA, self).__init__()
        self.n_feats = opt.n_features
        self.center = opt.past_frames
        self.num_ff = opt.future_frames
        self.num_fb = opt.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        # self.related_f = self.num_ff + self.num_fb
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(opt.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs)
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4, 160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)
                cor = self.F_p(cor)
                cor = self.condense(w * cor)
                cor_l.append(cor)
        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))

        return out

# RDB-based RNN cell
class RDBCell(nn.Module):
    def __init__(self, opt):
        super(RDBCell, self).__init__()
        self.n_channels = opt.n_channels
        self.activation = opt.activation
        self.n_feats = opt.n_features
        self.n_blocks = opt.n_blocks
        self.F_B0 = conv5x5(self.n_channels, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)
        self.F_R = RDNet(in_chs=(1 + 4) * self.n_feats, growth_rate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)  # in: 80
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3((1 + 4) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(self.n_feats, self.n_feats)
        )

    def forward(self, x, s_last):
        out = self.F_B0(x)
        out = self.F_B1(out)
        out = self.F_B2(out)
        out = torch.cat([out, s_last], dim=1)
        out = self.F_R(out)
        s = self.F_h(out)

        return out, s
        
# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, opt):
        super(Reconstructor, self).__init__()
        self.opt = opt
        self.num_ff = opt.future_frames
        self.num_fb = opt.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        # self.related_f = self.num_ff + self.num_fb
        self.n_feats = opt.n_features
        self.n_channels = opt.n_channels
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, self.n_channels, stride=1)
        )

    def forward(self, x):
        return self.model(x)

# ESTRNN Model
class ESTRNNModel(nn.Module):
    """
    Efficient saptio-temporal recurrent neural network (ESTRNN, ECCV2020)
    """
    def __init__(self, opt):
        super(ESTRNNModel, self).__init__()
        self.opt = opt
        self.n_feats = opt.n_features
        self.num_ff = opt.future_frames
        self.num_fb = opt.past_frames
        self.ds_ratio = 4
        self.device = opt.device
        self.cell = RDBCell(opt)
        self.recons = Reconstructor(opt)
        self.fusion = GSA(opt)

    def forward(self, x, profile_flag=False):
        if profile_flag:
            return self.profile_forward(x)
        outputs, hs = [], []
        # print('x.shape:', x.shape)
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        # forward h structure: (batch_size, channel, height, width)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s = self.cell(x[:, i, :, :, :], s)
            hs.append(h)
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)

    # For calculating GMACs
    def profile_forward(self, x):
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s = self.cell(x[:, i, :, :, :], s)
            hs.append(h)
        for i in range(self.num_fb + self.num_ff):
            hs.append(torch.randn(*h.shape).to(self.device))
        for i in range(self.num_fb, frames + self.num_fb):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params