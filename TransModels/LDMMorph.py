import torch.nn as nn
import torch
import TransModels.Conv2dReLU as Conv2dReLU
import TransModels.LWSA as LWSA
import TransModels.LWCA as LWCA 
import TransModels.Decoder as Decoder
import utils.configs as configs

class LDMMorph(nn.Module): 
    def __init__(self, channel_1, channel_2, channel_3, channel_4):
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3
        self.channel_4 = channel_4

        super(LDMMorph, self).__init__()

        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.ec1 = Conv2dReLU.Conv2dReLU(2, 32, 3, 1, use_batchnorm=False)
        self.start_channel = 64
        bias_opt = True


        config1 = configs.get_SelfAttention_config()  
        config2 = configs.get_CrossAttention_config()

        self.lwsa = LWSA.LWSA(config1, in_channel=2)
        
        self.c1 = self.encoder(self.channel_1, self.start_channel,     bias=bias_opt)  #->size32
        self.c2 = self.encoder(self.channel_2, self.start_channel * 2, bias=bias_opt)  #->size16
        self.c3 = self.encoder(self.channel_3, self.start_channel * 4, bias=bias_opt)  #->size8
        self.c4 = self.encoder(self.channel_4, self.start_channel * 8, bias=bias_opt)  #->size4
        
        self.lwca1 = LWCA.LWCA(config2, dim_diy=64)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=128)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=256)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=512)

        self.up0 = Decoder.DecoderBlock(512, 256, skip_channels=256, use_batchnorm=False)
        self.up1 = Decoder.DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)
        self.up2 = Decoder.DecoderBlock(128, 64, skip_channels=64, use_batchnorm=False)
        self.up3 = Decoder.DecoderBlock(64, 32, skip_channels=32, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        
        self.reg_head = Decoder.RegistrationHead(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
        )
    
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
    
    def forward(self, moving_Input, fixed_Input, score1, score2, score3, score4):

        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)

        x_s1 = self.avg_pool(input_fusion)  
        f4 = self.ec1(x_s1)

        swin_fea_4, swin_fea_8, swin_fea_16, swin_fea_32 = self.lwsa(input_fusion)
        cnn_fea_4, cnn_fea_8, cnn_fea_16, cnn_fea_32 = self.c1(score1), self.c2(score2), self.c3(score3), self.c4(score4)

        moving_fea_4_cross = self.lwca1(swin_fea_4, cnn_fea_4)
        moving_fea_8_cross = self.lwca2(swin_fea_8, cnn_fea_8)
        moving_fea_16_cross = self.lwca3(swin_fea_16, cnn_fea_16)
        moving_fea_32_cross = self.lwca4(swin_fea_32, cnn_fea_32)

        fixed_fea_4_cross = self.lwca1(cnn_fea_4, swin_fea_4)
        fixed_fea_8_cross = self.lwca2(cnn_fea_8, swin_fea_8)
        fixed_fea_16_cross = self.lwca3(cnn_fea_16, swin_fea_16)


        x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)
        x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)
        x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)

        v = self.reg_head(x)

        return v



