import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch

from dataset import get_dataloader
from model import Triplet_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('t-SNE / UMAP Visualization of Encoder Features')
    parser.add_argument('--dataset', type=str, default="/home/admin-ps/Documents/hatAndMask/SignatureDatasets/ChiSig")
    parser.add_argument('--model_path', type=str, default="./saved_models/DTL_ChiSig_backbone=None_RNone_C_self.pt")
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--use_umap', action='store_true', help="Use UMAP instead of t-SNE")
    parser.add_argument('--learning_rate', type=float, default=1.0)
    parser.add_argument('--learning_rate_AE', type=float, default=0.005)
    args = parser.parse_args()

    # 1. 加载数据集 (使用 test_loader)
    _, test_loader = get_dataloader(args)

    # 2. 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    print(f"Loading model from: {args.model_path} | Epochs trained: {checkpoint['epochs']}")
    model = Triplet_Model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 3. 提取 encoder + projector 输出的特征
    features, labels, writer_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            feat = model.projector(model.encoder(batch['image'].to(device), pool=True))
            features.append(feat.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
            writer_ids.append(batch['writer_id'])

    # 4. 转换成 numpy
    features = np.array(features)  # (N, 1, feat_dim)
    labels = np.array(labels)
    writer_ids = np.array(writer_ids)

    n_samples = features.shape[0] * features.shape[1]
    n_features = features.shape[2]
    features = features.reshape(n_samples, n_features)
    labels = labels.reshape(labels.shape[0])

    print(f">> Extracted features shape: {features.shape}")
    # 观察数据标准差
    print(features.min(), features.max(), features.std())

    # 5. 降维
    if args.use_umap:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        features_2d = reducer.fit_transform(features)
        method = "UMAP"
    else:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
        features_2d = tsne.fit_transform(features)
        method = "t-SNE"

    # 6. 绘制真伪分布
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label='Genuine' if label == 1 else 'Forged',
                    alpha=0.6, s=20)
    plt.legend()
    plt.title(f"{method} Visualization of Encoder Features")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.savefig(f"visualization_{method}.png", dpi=300)
    plt.show()

    print(f">> {method} 可视化保存到 visualization_{method}.png")
