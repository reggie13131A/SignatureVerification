import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW

from Network_modules import *
from optimizer import *
from scheduler import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(1)

class SSL_Model(nn.Module):
    def __init__(self, args):
        super(SSL_Model, self).__init__() 

        self.args = args
        self.encoder = InceptionResNet_V2(self.args.is_pretrained)
        self.attn_module = Attn2D_Module()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = UNet_Decoder()
        self.train_params = self.parameters()

        # 参数group, setting different kinds of learning rate
        self.encoder_params, self.new_params = [], []
        for name, param in self.named_parameters():
            if "encoder" in name or "backbone" in name:   
                self.encoder_params.append(param)
            else:  
                self.new_params.append(param)

        # if args.optimizer.upper() == 'SGD':
        #     self.optimizer = SGD(self.train_params, lr=self.args.learning_rate)
        # elif args.optimizer.upper() == 'LARS':
        #     self.optimizer = LARS(self.train_params, self.args.learning_rate)

        if args.optimizer.upper() == 'SGD':
            self.optimizer = SGD([
                {"params": self.encoder_params, "lr": self.args.Encoder_learning_rate},
                {"params": self.new_params, "lr": self.args.learning_rate}
            ])
        elif args.optimizer.upper() == 'LARS':
            self.optimizer = LARS([
                {"params": self.encoder_params, "lr": self.args.Encoder_learning_rate},
                {"params": self.new_params, "lr": self.args.learning_rate}
            ])
        else: # Adam
            self.optimizer = AdamW([
                {"params" : self.encoder_params,  "lr" : self.args.Encoder_learning_rate},
                {"params" : self.new_params, "lr" : self.args.learning_rate},
            ], self.args.learning_rate, weight_decay=0.0005)
        # self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, self.args.warmup_epochs, self.args.max_epochs)

    # def train_SSL(self, batch):
    #     print("Initial learning rate:", self.optimizer.param_groups[0]['lr'])
    #     self.train()
    #     self.optimizer.zero_grad()
    #     reconstr_loss = 0.0   

    #     '''
    #     TODO:   1. iterate through batch and perform: i) Attn ii) Reconstruction
    #             2. return reconstructed images at epoch intervals
    #             [N,P,C,H,W] --(+)--> [N,P,512] --> [N,512,4,4] --> [N,512] --> decoder
    #     '''

    #     batch['image'] = batch['image'].to(device)      # [N,3,256,256] 
    #     print(f"batch['image'].shape: {batch['image'].shape}")
    #     batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64]
    #     print(f"batch['patches'].shape: {batch['patches'].shape}")

    #     image_feature = self.encoder(batch['image'], pool=True) # [N,512]

    #     patch_attn_feats = []
    #     patch_attn_maps = []
        
    #     ### for N = 1 ###
    #     """
    #         这里的设计是有讲究的，0dim的意思就是从batch中取出一个[16, 3, 64, 64]的patches
    #         然后遍历所有的patches，计算每一个patch的特征图
    #         如果要修改就得看能不能做到：输入一个batch，batch有32个patches，一个patches有16个patch，
    #         一次性将32个patch进行输入计算
    #     """
    #     for patch in batch['patches'][0]: 
    #     # patch = batch['patches'][0] # [64,3,32,32]
    #         patch_feature_map = self.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
    #         attn_map, attn_feature = self.attn_module(image_feature, patch_feature_map) # [N,512]
    #         patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]
    #         patch_attn_maps.append(attn_map)

    #     patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
    #     patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
        
    #     patch_attn = self.avg_pool(patch_attn_feats) # [N,512,1]
    #     patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]
    #     # patch_attn = patch_attn_feats.reshape(-1, patch_attn_feats.shape[1], int(math.sqrt(patch_attn_feats.shape[2])), int(math.sqrt(patch_attn_feats.shape[2]))) # [N,512,4,4]
        
    #     recons_image = self.decoder(patch_attn) # [N,3,256,256]

    #     # 计算的方式是mse loss，计算两个tensor之间的损失，一个是dataloader之下的原始image，一个是经过网络重新生成的图片
    #     reconstr_loss += F.mse_loss(batch['image'], recons_image)
    #     print(f"损失的大小是：{ reconstr_loss.item() }")
    #     # 这其实也是一个降损的过程
    #     reconstr_loss.backward()
    #     self.optimizer.step()

    #     image_results = torch.cat([batch['image'].to(device), recons_image.to(device)], dim=0)
    #     return patch_attn_maps, reconstr_loss.item(), image_results

    def train_SSL(self, batch):
        self.train()
        self.optimizer.zero_grad()

        # 数据转移到设备
        batch['image'] = batch['image'].to(device)      # [N,3,256,256] N=32
        batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64] N=32, 16个patch

        # 编码完整图像 - 一次性处理32张图像
        image_feature = self.encoder(batch['image'], pool=True)  # [32,512]

        N, P = batch['patches'].shape[:2]  # N=32(batch_size), P=16(num_patches)

        patch_attn_feats = []
        patch_attn_maps = []

        # 按论文架构：迭代16次，每次处理所有32个样本的第i个patch
        for patch_idx in range(P):  # 迭代16次
            # 取出所有32张图像的第patch_idx个patch
            current_patches = batch['patches'][:, patch_idx, :, :, :]  # [32,3,64,64]
            # print(f"Processing patch {patch_idx}, current_patches shape: {current_patches.shape}")

            # 一次性编码32个相同位置的patch
            patch_feature_maps = self.encoder(current_patches, pool=False)  # [32,512,2,2] ---> 512个特征图
            # print(f"patch_feature_maps shape: {patch_feature_maps.shape}")

            # 检查并处理encoder输出维度
            if len(patch_feature_maps.shape) == 4:
                # attention model期待输入的 dim=4
                patch_features_4d = patch_feature_maps  # [32,512,h,w]
            else:
                # 如果已经是2D，转换为4D
                patch_features_4d = patch_feature_maps.unsqueeze(-1).unsqueeze(-1)  # [32,512,1,1]

            # 批量计算注意力 - 32个image_feature与32个patch_feature做注意力
            # image_feature: [32,512] -> 可能需要reshape为[32,512,1,1]以匹配patch_features_4d
            if len(image_feature.shape) == 2:
                image_feature_4d = image_feature.unsqueeze(-1).unsqueeze(-1)  # [32,512,1,1]
            else:
                image_feature_4d = image_feature

            # 这里实现了"32个patch与32个img_fea做自注意力操作"
            attn_map, attn_feature = self.attn_module(
                image_feature,  # [32,512,1,1] 
                patch_features_4d  # [32,512,h,w] 或 [32,512,1,1]
            )

            # 存储结果
            patch_attn_feats.append(attn_feature.unsqueeze(1))  # [32,1,512]
            patch_attn_maps.append(attn_map)

        # 合并所有16个patch的特征
        patch_attn_feats = torch.cat(patch_attn_feats, dim=1)  # [32,16,512]
        patch_attn_feats = patch_attn_feats.permute(0, 2, 1)   # [32,512,16]

        # 特征聚合
        patch_attn = self.avg_pool(patch_attn_feats)  # [32,512,1]
        patch_attn = patch_attn.permute(0, 2, 1).squeeze(1)  # [32,512]

        # 图像重建 - 32张图像同时重建
        recons_image = self.decoder(patch_attn)  # [32,3,256,256]
        print(f"最终的重建大小是：{recons_image.shape}")

        # 计算损失
        reconstr_loss = F.mse_loss(batch['image'], recons_image)
        print(f"损失的大小是：{reconstr_loss.item()}")
                # 检查梯度是否存在
        print(f"损失是否需要梯度: {reconstr_loss.requires_grad}")

        # 反向传播
        reconstr_loss.backward()
        total_norm = 0
        grad_count = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_count += 1
            else:
                print(f"警告: 参数 {name} 没有梯度!")

        if grad_count > 0:
            total_norm = total_norm ** (1. / 2)
        else:
            print("错误: 没有任何参数有梯度!")

        # 学习率检查
        print(f"当前学习率: {self.optimizer.param_groups[0]['lr']}")

        self.optimizer.step()

        # 返回结果
        image_results = torch.cat([
            batch['image'].to(device), 
            recons_image.to(device)
        ], dim=0)  # [64,3,256,256] - 原图32张+重建图32张

        return patch_attn_maps, reconstr_loss.item(), image_results

    def get_attn_recons(self, batch):
        '''No gradients are to flow'''
        self.eval()
        with torch.no_grad():
            batch['image'] = batch['image'].to(device)      # [N,3,256,256] 
            batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64]

            image_feature = self.encoder(batch['image'], pool=True) # [N,512]

            patch_attn_feats = []
            patch_attn_maps = []
            
            ### for N = 1 ###
            for patch in batch['patches'][0]: 
            # patch = batch['patches'][0] # [64,3,32,32]
                patch_feature_map = self.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
                attn_map, attn_feature = self.attn_module(image_feature, patch_feature_map) # [N,512]
                patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]
                # patch_attn_maps.append(attn_map)

                # visualise attention maps #
                # print(attn_map.shape)
                attn_map = attn_map.unsqueeze(-1).view(attn_map.shape[0], 1, 2, 2) # [N,1,4] -> [N,1,2,2]
                I_grid = make_grid(batch['image'].to(device), normalize=True, scale_each=True)
                imposed_attn = visualize_attn_softmax(I_grid, attn_map, up_factor=128)
                # print(f"superimposed attn: {imposed_attn.shape}")
                patch_attn_maps.append(imposed_attn.unsqueeze(0)) 

            # print(np.array(patch_attn_maps).shape)
            patch_attn_maps = torch.mean(torch.stack(patch_attn_maps), dim=0)
            # print(f"patch attn map: {patch_attn_maps.shape}")
            patch_attn_maps = torch.cat([patch_attn_maps.to(device)], dim=0)

            patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
            patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
            
            patch_attn = self.avg_pool(patch_attn_feats) # [N,512,1]
            patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]
            
            recons_image = self.decoder(patch_attn) # [N,3,256,256]
        
        return patch_attn_maps, recons_image






