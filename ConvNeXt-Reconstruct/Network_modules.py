import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# 不要用timm这个库，因为它是hugging face的产品，随时把中国用户给ban了
import timm 
import pretrainedmodels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(1)

# 我觉得这个可以更改
# class RN18_Encoder(nn.Module):
#     def __init__(self, pretrained=False):
#         super(RN18_Encoder, self).__init__()
#         # 这里用的是一个resnet18的预训练模型
#         self.is_pretrained = pretrained
#         # 这里直接把resnet18给加载进来了
#         self.backbone = torchvision.models.resnet18(pretrained=self.is_pretrained)

#         # to get feature map -- model without 'avgpool' and 'fc' --- 相当于把conv、残差层那些全给加到nn.sequential中
#         self.features = nn.Sequential()
#         for name, module in self.backbone.named_children():
#             if name not in ['avgpool', 'fc']:
#                 self.features.add_module(name, module)
#         self.gap = nn.AdaptiveAvgPool2d(1)

#     def forward(self, x, pool=True):
#         x = self.features(x)
#         if pool:
#             x = self.gap(x)
#             x = torch.flatten(x, 1)
#         return x

class InceptionResNet_V2(nn.Module):
    def __init__(self, pretrained=False, freeze_stem=False):
        super(InceptionResNet_V2, self).__init__()
        # 加载 inceptionresnetv2
        if pretrained:
            self.backbone = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained='imagenet')
        else:
            self.backbone = pretrainedmodels.__dict__['inceptionresnetv2'](pretrained=None)

        self.features = self.backbone.features

        # 可选：冻结 stem 部分（前几层）
        if freeze_stem:
            for name, module in self.features.named_children():
                if name in ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'maxpool_3a', 'conv2d_3b', 'conv2d_4a', 'mixed_5b']:
                    for p in module.parameters():
                        p.requires_grad = False

        # 可选的全局池化
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, pool=True):
        x = self.features(x)  # [B, 1536, H', W']
        if pool:
            x = self.gap(x)  # [B, 1536, 1, 1]
            x = torch.flatten(x, 1)
        return x


class Attn2D_Module(nn.Module):
    def __init__(self):
        super(Attn2D_Module, self).__init__()
        self.feature_dim = 1536
        # self.feature_dim = 512
        self.embedding_size = 1024

        # 线性层用来维度转换
        self.conv_h = nn.Linear(self.feature_dim, self.embedding_size)
        # 一个卷积层
        self.conv_f = nn.Conv2d(self.feature_dim, self.embedding_size, kernel_size=3, padding=1)
        # 这里的维度转换很剧烈，所以使用多头注意力机制
        # self.conv_att = nn.Linear(self.embedding_size, 1)
        self.num_heads = 4
        self.conv_att = nn.ModuleList([
            nn.Linear(self.embedding_size, 1) for _ in range(self.num_heads)
        ])
        
    # 这段函数实现的基本是论文中的图了，参数1只是一个线性变换，参数2参与卷积，这里跟论文的描述不符
    def forward(self, img_fvec, patch_fmap):
        g_em = self.conv_h(img_fvec) # [N,1536] -> [N, 1024]
        g_em = g_em.unsqueeze(-1).permute(0,2,1) # [N, 1024] -> [N,1024,1] -> [N,1,1024]

        x_em = self.conv_f(patch_fmap) # [N,1536,2,2] -> [N,256,2,2]
        x_em = x_em.view(x_em.shape[0],-1,x_em.shape[2]*x_em.shape[3]).permute(0,2,1) # [N,256,8,8] -> [N,256,64] -> [N,64,256]

        actv_sum_feat = torch.tanh(x_em + g_em) # [N,64,256]
        scores = [att(actv_sum_feat) for att in self.conv_att]
        scores = torch.cat(scores, dim=-1).mean(dim=-1, keepdim=True)
        # attn_wts = F.softmax(self.conv_att(actv_sum_feat), dim=1).permute(0,2,1) # [N,64,256] -> [N,64,1] -> [N,1,64]
        attn_wts = F.softmax(scores, dim=1).permute(0,2,1) # [N,64,256] -> [N,64,1] -> [N,1,64]

        patch_fmap_ = patch_fmap.view(patch_fmap.shape[0],-1,patch_fmap.shape[2]*patch_fmap.shape[3]) # [N,512,8,8] -> [N,512,64]
        patch_fmap_ = patch_fmap_.permute(0,2,1) # [N,512,64] -> [N,64,512]

        attn_output = torch.bmm(attn_wts, patch_fmap_) # [N,1,64] x [N,64,512] => [N,1,512]
        attn_output = attn_output.squeeze(1) # [N,1,512] -> [N,512]

        return attn_wts, attn_output


# 一个上采样的网络
class Decoder(nn.Module):
    def __init__(self, fmap_dims=8, out_channels=3):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.fmap_dims = fmap_dims
        self.num_layers = int(math.log2(self.fmap_dims))
        self.conv_layers = []
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # upsampling layers
        for i in range(self.num_layers):
            self.conv_layers.append(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1))

        self.upconv_dims = nn.Sequential(*self.conv_layers)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_image = nn.ConvTranspose2d(32, self.out_channels, kernel_size=4, stride=2, padding=1)

    # 接收一段 [N,512] 的tensor，解码它并且还原一张图片出来
    def forward(self, x):
        x = x.view(-1,512,1,1)
        # print(x.shape)
        # 经历三次相同channel下的上采样
        x = self.upconv_dims(x)
        # print(x.shape)
        x = self.relu(self.upconv1(x))
        # print(x.shape)
        x = self.relu(self.upconv2(x))
        # print(x.shape)
        x = self.relu(self.upconv3(x))
        # print(x.shape)
        x = self.relu(self.upconv4(x))
        # print(x.shape)
        x = self.final_image(x)
        # print(x.shape)
        return x

# 接收一段 [N,512] 的tensor，解码它并且还原一张图片出来
class UNet_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNet_Decoder, self).__init__()
        self.deconv_1 = Unet_UpBlock(1536, 768)
        self.deconv_2 = Unet_UpBlock(768, 512)
        self.deconv_3 = Unet_UpBlock(512, 512)
        self.deconv_4 = Unet_UpBlock(512, 256)
        self.deconv_5 = Unet_UpBlock(256, 128)
        self.deconv_6 = Unet_UpBlock(128, 64)
        self.deconv_7 = Unet_UpBlock(64, 32)
        self.final_image = nn.Sequential(*[nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)])

    def forward(self, x):
        print(f"经过decoder的输入形状是{x.shape}")
        x = x.view(-1, 1536, 1, 1)
        x = self.deconv_1(x)  # 2
        x = self.deconv_2(x)  # 4
        x = self.deconv_3(x)  # 8
        x = self.deconv_4(x)  # 16
        x = self.deconv_5(x)  # 32
        x = self.deconv_6(x)  # 64
        x = self.deconv_7(x)  # 128
        x = self.final_image(x)  # 256
        return x


class Unet_UpBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc):
        super(Unet_UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(outer_nc),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder_4x4(nn.Module):
    def __init__(self, fmap_dims=2, out_channels=3):
        super(Decoder_4x4, self).__init__()
        self.out_channels = out_channels
        self.fmap_dims = fmap_dims
        self.num_layers = int(math.log2(self.fmap_dims))
        self.conv_layers = []
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # upsampling layers
        for i in range(self.num_layers):
            self.conv_layers.append(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1))

        self.upconv_dims = nn.Sequential(*self.conv_layers)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_image = nn.ConvTranspose2d(32, self.out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x = x.view(-1,512,1,1)
        print(x.shape)
        x = self.upconv_dims(x)
        print(x.shape)
        x = self.relu(self.upconv1(x))
        print(x.shape)
        x = self.relu(self.upconv2(x))
        print(x.shape)
        x = self.relu(self.upconv3(x))
        print(x.shape)
        x = self.relu(self.upconv4(x))
        print(x.shape)
        x = self.final_image(x)
        print(x.shape)
        return x



if __name__ == '__main__':
    # model_enc = RN18_Encoder()
    # for img_size in [32, 64, 128, 256]:
    #     print(f"\n>>Image/Patch size = {img_size}x{img_size}")
    #     img_tns = torch.randn((4,3,img_size,img_size))
    #     fmap = model_enc(img_tns, pool=False)
    #     model_dec = Decoder(fmap_dims=fmap.shape[-1])
    #     fvec = model_enc(img_tns, pool=True)
    #     fimg = model_dec(fvec)
    #     print(fmap.shape, fvec.shape, fimg.shape)
    # patch_fmap = torch.randn((10,512,2,2))
    # img_fvec = torch.randn((10,512))
    # attn_map, attn_out = Attn2D_Module().forward(img_fvec, patch_fmap)
    # print(attn_map.shape, attn_out.shape)
    fmap = torch.randn(4,512,4,4)
    model_dec = Decoder_4x4()
    fimg = model_dec(fmap)
    print(fimg.shape)