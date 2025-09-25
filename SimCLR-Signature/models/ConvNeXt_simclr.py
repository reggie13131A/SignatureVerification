import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXtSimCLR(nn.Module):
    def __init__(self, variant='convnext_v2_large', out_dim=1536):
        super().__init__()

        # 获取 torchvision 中的 ConvNeXtV2 模型（不加载预训练）
        backbone_fn = getattr(models, variant)
        backbone = backbone_fn(pretrained=False)

        # 取出 embedding 维度
        dim_mlp = backbone.classifier[2].in_features

        # 去掉分类头
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp, bias=False),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)                          # (B, C, H, W)
        h = nn.functional.adaptive_avg_pool2d(h, 1)   # (B, C, 1, 1)
        h = torch.flatten(h, 1)                       # ✅ (B, C)
        z = self.projector(h)                         # (B, out_dim)
        return z

# 测试
model = ConvNeXtSimCLR(variant='convnext_large', out_dim=1536)
model = model.to('cuda')
x = torch.randn(32, 3, 256, 256).to('cuda')
z = model(x)
print(z.shape)  # ✅ 正确输出应该是 (32, 1536)
