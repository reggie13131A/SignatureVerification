import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pretrainedmodels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1)


# class RN18_Encoder(nn.Module):
#     def __init__(self, pretrained=False):
#         super(RN18_Encoder, self).__init__()
#         self.is_pretrained = pretrained
#         self.backbone = torchvision.models.resnet18(pretrained=self.is_pretrained)

#         # to get feature map -- model without 'avgpool' and 'fc'
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


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_image = nn.ConvTranspose2d(32, self.out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x = x.view(-1,512,1,1)
        # # print(x.shape)
        # x = self.upconv_dims(x)
        # # print(x.shape)
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


class UNet_Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(UNet_Decoder, self).__init__()
        # self.linear_1 = nn.Linear(512, 8*8*256)
        # self.dropout = nn.Dropout(0.5)
        # self.deconv_1 = Unet_UpBlock(512, 512)
        # self.deconv_2 = Unet_UpBlock(512, 512)
        # self.deconv_3 = Unet_UpBlock(512, 512)
        self.deconv_4 = Unet_UpBlock(512, 256)
        self.deconv_5 = Unet_UpBlock(256, 128)
        self.deconv_6 = Unet_UpBlock(128, 64)
        self.deconv_7 = Unet_UpBlock(64, 32)
        self.final_image = nn.Sequential(*[nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)])

    def forward(self, x):
        # x = self.linear_1(x)
        # x = x.view(-1, 512, 1, 1)
        # x = self.dropout(x)
        # x = self.deconv_1(x)  # 2
        # x = self.deconv_2(x)  # 4
        # x = self.deconv_3(x)  # 8
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


if __name__ == '__main__':
    M_enc = RN18_Encoder()
    M_dec = UNet_Decoder()
    img = torch.rand((4,3,256,256))
    print(img.shape)
    fmap = M_enc(img, pool=False)
    print(fmap.shape)
    recons = M_dec(fmap)
    print(recons.shape)