import torch.nn.functional as F
import torch.nn as nn
import torch


def process(t):
    return t.squeeze()

def per_err(x, error):
    perceptual_errormap = x * error
    # print('p', perceptual_errormap.shape)
    perceptual_errormap = perceptual_errormap[:, 4:-4, 4:-4]
    return perceptual_errormap

def totalVari_regu(senMap, beta=3):

    sobel_h = torch.Tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
    sobel_h = sobel_h.unsqueeze(0)
    sobel_h = sobel_h.unsqueeze(0)
    sobel_h = sobel_h.cuda()
    sobel_w = torch.Tensor([[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]])
    sobel_w = sobel_w.unsqueeze(0)
    sobel_w = sobel_w.unsqueeze(0)
    sobel_w = sobel_w.cuda()

    if len(senMap.shape) == 3:
        senMap = senMap.unsqueeze(1)

    h = F.conv2d(senMap, sobel_h, bias=None, stride=1, padding=1)
    w = F.conv2d(senMap, sobel_w, bias=None, stride=1, padding=1)

    tv = (h**2+w**2)**(beta/2.)

    tv = F.adaptive_avg_pool2d(tv, output_size=(1, 1)).squeeze()

    return tv

