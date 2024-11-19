'''
Jiong Wu 
University of Florida
jiongwu.application@ufl.edu

Thanks to 
Junyu Chen
Johns Hopkins Unversity
jchen245@jhmi.edu
'''

import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import torch.utils.data as Data
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0
  
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        h2, w2 = mov_image.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]

        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True,padding_mode="border")
        
        return sample_grid, warped


def smoothloss(y_pred):
    h2, w2 = y_pred.shape[-2:]
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0


def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff


class MSE:
    """
    Mean squared error loss.
    """
 
    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def crop_center(img, cropx, cropy, cropz):
    x, y, z = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    startz = z//2 - cropz//2
    return img[startx:startx+cropx, starty:starty+cropy, startz:startz+cropz]


def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min)/(i_max - i_min)
    return norm

def loadnpz(npzpath):
    features=np.load(npzpath, allow_pickle=True)
    f_all = features['arr_0'].item()
    imglist = f_all['imglist']
    movimg = imglist[0,:,:]
    movlab = imglist[1,:,:]
    tarimg = imglist[2,:,:]
    tarlab = imglist[3,:,:]
    return movimg, movlab, tarimg, tarlab

class Dataset_epoch_with_name(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names):
        'Initialization'
        super(Dataset_epoch_with_name, self).__init__()
        self.names = names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, index):
        'Generates one sample of data'
        arr = np.load(self.names[index])

        movimg = imgnorm(arr["img_small"])
        tarimg = imgnorm(arr["img_large"])
        movlab = arr["mask_small"]
        tarlab = arr["mask_large"]

        movimg = torch.from_numpy(movimg).float()
        tarimg = torch.from_numpy(tarimg).float()
        movlab = torch.from_numpy(movlab).float()
        tarlab = torch.from_numpy(tarlab).float()

        pairname = self.names[index].split('/')[-1].split('.npz')[0]

        return movimg.unsqueeze(0), tarimg.unsqueeze(0), movlab.unsqueeze(0), tarlab.unsqueeze(0), pairname


class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names):
        'Initialization'
        super(Dataset_epoch, self).__init__()
        self.names = names

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, index):
        'Generates one sample of data'
        npzpath = self.names[index]
        movimg, movlab, tarimg, tarlab = loadnpz(npzpath)

        movimg = torch.from_numpy(movimg).float()
        tarimg = torch.from_numpy(tarimg).float()
        movlab = torch.from_numpy(movlab).float()
        tarlab = torch.from_numpy(tarlab).float()

        return movimg.unsqueeze(0), tarimg.unsqueeze(0), movlab.unsqueeze(0), tarlab.unsqueeze(0)

