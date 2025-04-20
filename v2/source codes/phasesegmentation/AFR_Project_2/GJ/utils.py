import numpy as np
#import matplotlib.pyplot as plt
import torch


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp1 = std * inp + mean
    inp1 = np.clip(inp1, 0, 1)
    ##plt.imshow(inp1)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

    return (np.uint8(inp1))

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # if one_channel:
    #     plt.imshow(npimg, cmap="Greys")
    # else:
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return npimg

def show_img(img):
    # unnormalize the images
    img = img * 0.22 + 0.45
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    img1 = np.clip(npimg, 0., 1.)

    return img1 # return the unnormalized images

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']