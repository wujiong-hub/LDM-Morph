'''
LWSA module

A partial code was retrieved from:
https://github.com/tzayuan/TransMatch_TMI and
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
'''

import TransModels.basic_LWSA as basic
import TransModels.Conv2dReLU as Conv2dReLU 
import torch.nn as nn
import utils.configs as configs

class LWSA(nn.Module):
    def __init__(self, config, in_channel=1):
        super(LWSA, self).__init__() 
        if_convskip = config.if_convskip 
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = basic.SwinTransformer(patch_size=config.patch_size,
                                                in_chans=config.in_chans,
                                                embed_dim=config.embed_dim,
                                                depths=config.depths,
                                                num_heads=config.num_heads,
                                                window_size=config.window_size,
                                                mlp_ratio=config.mlp_ratio,
                                                qkv_bias=config.qkv_bias,
                                                drop_rate=config.drop_rate,
                                                drop_path_rate=config.drop_path_rate,
                                                ape=config.ape,
                                                patch_norm=config.patch_norm,
                                                use_checkpoint=config.use_checkpoint,
                                                out_indices=config.out_indices
                                           )
        self.c1 = Conv2dReLU.Conv2dReLU(in_channel, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv2dReLU.Conv2dReLU(in_channel, config.reg_head_chan, 3, 1, use_batchnorm=False)

        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        source = x
        if self.if_convskip:
            x_s0 = x.clone()  
            x_s1 = self.avg_pool(x) 
            f4 = self.c1(x_s1)  
            f5 = self.c2(x_s0) 
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        return f3, f2, f1, out_feats[-1]
