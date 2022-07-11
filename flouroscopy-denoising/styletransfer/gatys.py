from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imageio
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    image = imageio.imread(image_name)
    h, w = image.shape
    image = image.reshape(1, 1, h, w)
    image = torch.from_numpy(image).to(device)

    return image

def image_save(image, name):
    img = image.squeeze().detach().to('cpu').numpy()
    imageio.imsave(name, img)

def concat_image(*imgs):
    concat_img = []
    for i in imgs:
        # img = i.squeeze().detach().to('cpu').numpy()
        concat_img.append(i)

    concat_img = torch.cat(concat_img, dim=3)
    # print('concat_img.shape:', concat_img.shape)

    return concat_img


def unloader(img_tensor):
    img_np = img_tensor.detach().to('cpu').numpy()
    return img_np

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze()      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    plt.show()

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    if style_img.size(1) == 1:
        style_img = torch.cat([style_img] * 3, dim=1)
    if content_img.size(1) == 1:
        content_img = torch.cat([content_img] * 3, dim=1)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=100,
                    #    style_weight=1000000, content_weight=1):
                       style_weight=100000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

# if __name__ == '__main__':
#     style_path = r'../../../data/flouroscopy-denoising/test-results/moving700-20210313-1921-mfcnn2n-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/moving700/rotate_set3/out-rotate_set3-045.tiff'
#     content_path = r'../../../data/flouroscopy-denoising/test-results/moving700-20210316-0319-mfcnn2n2-n_inputs5-ms_channels32-growth_rate32-n_denselayers5-perceptual_loss-perceptual_loss/moving700/rotate_set3/out-rotate_set3-045.tiff'

#     style_img = image_loader(style_path)
#     content_img = image_loader(content_path)

#     print('style_img.shape::', style_img.shape)
#     print('content_img.shape:', content_img.shape)

#     assert style_img.size() == content_img.size(), \
#         "we need to import style and content images of the same size"

#     # plt.figure()
#     # imshow(style_img, title='Style Image')

#     # plt.figure()
#     # imshow(content_img, title='Content Image')

#     cnn = models.vgg19(pretrained=True).features.to(device).eval()

#     cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

#     content_layers_default = ['conv_4']
#     style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


#     input_img = content_img.clone()
#     # if you want to use white noise instead uncomment the below line:
#     # input_img = torch.randn(content_img.data.size(), device=device)

#     # add the original input image to the figure:
#     # plt.figure()
#     # imshow(input_img, title='Input Image')

#     output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                                 content_img, style_img, input_img, num_steps=1000)

#     image_save(output, 'output.tiff')
#     print('output.shape:', output.shape)
#     plt.figure()
#     imshow(output, title='Output Image')

#     cat_img = concat_image(*[style_img, content_img, output])
#     plt.figure()
#     imshow(cat_img, title='Concanate Image')
#     # sphinx_gallery_thumbnail_number = 4
#     plt.ioff()
#     plt.show()

def main():
    result_dir = r'../../../data/flouroscopy-denoising/style-transfer/results'
    compare_dir = r'../../../data/flouroscopy-denoising/style-transfer/compare'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    style_dir = r'../../../data/flouroscopy-denoising/test-results/moving700-20210313-1921-mfcnn2n-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/moving700'
    content_dir = r'../../../data/flouroscopy-denoising/test-results/moving700-20210316-0319-mfcnn2n2-n_inputs5-ms_channels32-growth_rate32-n_denselayers5-perceptual_loss-perceptual_loss/moving700'

    low_dir = r'../../../data/flouroscopy-denoising/test/moving700/low'
    gt_dir = r'../../../data/flouroscopy-denoising/test/moving700/high'

    style_list = sorted(glob.glob(os.path.join(style_dir, '*set*', '*.tiff')))
    content_list = sorted(glob.glob(os.path.join(content_dir, '*set*', '*.tiff')))
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    for sp, cp in zip(style_list, content_list):
        sf = os.path.basename(sp)
        cf = os.path.basename(cp)

        idx = sf.split('.')[-2].split('-')[-1]

        style_set = os.path.abspath(sp).split('\\')[-2]
        content_set = os.path.abspath(cp).split('\\')[-2]
        # gt_set_dir = os.path.

        # print(style_set, content_set)
        # print(sf, cf, idx)
        print(style_set, sf)
        gt_path = os.path.join(gt_dir, style_set, idx+'.tiff')
        low_path = os.path.join(low_dir, style_set, idx+'.tiff')
        
        style_img = image_loader(sp)
        content_img = image_loader(cp)
        low_img = image_loader(low_path)
        gt_img = image_loader(gt_path)

        print('style_img.shape::', style_img.shape)
        print('content_img.shape:', content_img.shape)
        print('low_img.shape:', low_img.shape)
        print('gt_img.shape:', gt_img.shape)

        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"

        assert cf == sf, "check wheter we loaded the same file name"
        assert style_set == content_set, "check whether we loaded the same set of images"

        os.makedirs(os.path.join(result_dir, style_set), exist_ok=True)
        os.makedirs(os.path.join(compare_dir, style_set), exist_ok=True)

        input_img = content_img.clone()
        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, num_steps=300) #default num_steps = 300
        
        cat_img = concat_image(*[low_img, style_img, content_img, output, gt_img])
        # avg_low = (low_img + content_img) / 2
        # sub_low = (low_img - content_img)
        # cat_img = concat_image(*[low_img, style_img, content_img, output, gt_img, avg_low, sub_low])

        result_path = os.path.join(result_dir, style_set, sf)
        compare_path = os.path.join(compare_dir, style_set, sf)

        image_save(output, result_path)
        image_save(cat_img, compare_path)

if __name__ == '__main__':
    main()