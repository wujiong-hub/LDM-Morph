import TransModels.Conv2dReLU as opt_Conv
import torch
from torch import nn
from torch.distributions.normal import Normal


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = opt_Conv.Conv2dReLU(
            in_channels + skip_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = opt_Conv.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = opt_Conv.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x, skip=None, skip2=None):
        x = self.up(x)
        if skip is not None:
            if skip.shape[2] != x.shape[2]:
                x = x[:,:,:skip.shape[2],:,:]
            x = torch.cat([x, skip], dim=1)
        if skip2 is not None:
            if skip2.shape[2] != x.shape[2]:
                x = x[:,:,:skip2.shape[2],:,:]
            x = torch.cat([x, skip2], dim=1)
            x = self.conv1(x)
        if skip2 is None:
            x = self.conv3(x)
        x = self.conv2(x)
        return x

class DecoderBlockPR(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = opt_Conv.Conv2dPReLU(
            in_channels + skip_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = opt_Conv.Conv2dPReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = opt_Conv.Conv2dPReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x, skip=None, skip2=None):
        x = self.up(x)
        if skip is not None:
            if skip.shape[2] != x.shape[2]:
                x = x[:,:,:skip.shape[2],:,:]
            x = torch.cat([x, skip], dim=1)
        if skip2 is not None:
            if skip2.shape[2] != x.shape[2]:
                x = x[:,:,:skip2.shape[2],:,:]
            x = torch.cat([x, skip2], dim=1)
            x = self.conv1(x)
        if skip2 is None:
            x = self.conv3(x)
        x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        sofsign = nn.Softsign()
        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))
        super().__init__(conv2d, sofsign)


