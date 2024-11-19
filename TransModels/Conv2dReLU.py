import torch.nn as nn

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm2d(out_channels)
        else:
            nm = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, nm, relu)

class Conv2dPReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        #relu = nn.LeakyReLU(inplace=True)
        relu = nn.PReLU()
        if not use_batchnorm:
            nm = nn.InstanceNorm2d(out_channels)
        else:
            nm = nn.BatchNorm2d(out_channels)

        super(Conv2dPReLU, self).__init__(conv, nm, relu)


class Conv2dEncode(nn.Sequential):
    def __init__(
             self,
             in_channels,
             start_channels,
             kernel_size=3,
             padding=0,
             strid=1,
             use_batchnorm=True,
     ):
        bias_opt = True
        self.in_channel = in_channels
        self.start_channel = start_channels
        super(Conv2dEncode, self).__init__()

        self.c1 = self.encoder(self.in_channel,    self.start_channel, stride=2, bias=bias_opt)  #48->96
        self.c2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)  #96->192
        self.c3 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt) #192->384
        self.c4 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt) #384->768

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        return x1, x2, x3, x4