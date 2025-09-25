import os
import numpy as np
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import torch
from dataset import *
from model import *
import warnings
from torchvision.transforms.functional import to_pil_image
warnings.filterwarnings('ignore')
from torchvision.utils import save_image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PreprocessAPic(Dataset):
    # 要求传入一个args
    def __init__(self, args, picPath):
        self.ptsz = args.ptsz
        self.basic_transforms = transforms.Compose([transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ])
        self.img_path = picPath

    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        image = image.convert('L')
        image = np.asarray(image)
        print("Image shape:", image.shape)  # 调试：检查输入图像尺寸
        print("Image min, max:", image.min(), image.max())  # 调试：检查像素值范围
        # 计算图像的质量中心

        if image.min() == image.max():
            print("Warning: Image is uniform (all white or all black), returning original")
            return Image.fromarray(image).convert('RGB')
        else:
            print("并不是一个全白或全黑的图像，继续处理")

        com = ndimage.measurements.center_of_mass(image)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        
        # 获取中心点的行和列
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :], image[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image[i, :])
            if v < 255*image.shape[1]:
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image[:, j])
            if v < 255*image.shape[0]: 
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        crop_rgb = Image.fromarray(np.asarray(image[x_start:x_end, y_start:y_end])).convert('RGB')
        return crop_rgb

    def __getpatches__(self, x_arr):
        patches = []
        C, H, W = x_arr.shape # 3, 256, 256

        num_H = H // self.ptsz
        num_W = W // self.ptsz

        for i in range(num_H):
            for j in range(num_W):
                start_x = i*self.ptsz
                end_x = start_x + self.ptsz
                start_y = j*self.ptsz
                end_y = start_y + self.ptsz

                patch = x_arr[:, start_x:end_x, start_y:end_y]
                patch_tns = torch.from_numpy(patch)
                patches.append(torch.unsqueeze(patch_tns, 0))

        return torch.cat(patches, dim=0)

    def __getitem__(self):
        sig_image = Image.open(self.img_path)
        sig_image.save("0_original.png")
        # 裁剪
        cropped_sig = self.__get_com_cropped__(sig_image)
        cropped_sig.save("1_cropped.png")
        sig_image = self.basic_transforms(cropped_sig)
        print("Transformed tensor range:", sig_image.min(), sig_image.max())
        sig_np = sig_image.numpy()
        sig_denorm = denormalize(sig_image.clone())  # 克隆避免修改原张量
        to_pil_image(sig_denorm).save("2_denormalized.png")

        sig_patches = self.__getpatches__(sig_np)
        sample = {'image' : sig_image, 'patches' : sig_patches}
        return sample


class PreprocessAll(Dataset):
    # 要求传入一个args
    def __init__(self, args, mode):

        self.args = args
        self.mode = mode
        self.ptsz = args.ptsz
        self.basic_transforms = transforms.Compose([transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ])

        self.augment_transforms = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.5),
                                                      transforms.RandomChoice([transforms.RandomEqualize(0.5),
                                                                               transforms.RandomInvert(0.5)]),
                                                      transforms.RandomApply([transforms.RandomAffine(
                                                          15, (0.05, 0.05), (1.5, 1.5), 0.5)], p=0.2)
                                                      ])

        # save all image paths with label (0/1)
        data_root = os.path.join(self.args.base_dir, self.args.dataset)  # BHSig260/Bengali
        print(f"目前的data_root是：{data_root}")

        data_df = pd.DataFrame(columns=['img_path', 'label'])

        for dir_name in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir_name)
            if os.path.isdir(dir_path):
                for img_name in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_name)
                    
                    # 新的标签判断逻辑：文件名中包含"-true"则标记为1，否则为0
                    label = 1 if '-true' in img_name else 0
                    data_df = data_df._append({'img_path': img_path, 'label': label}, ignore_index=True)

        print(f'{self.args.dataset} comprises total {len(data_df)} images !!')
        """
            下面这些是自己定义的，因为评估一般是输入n张图片作为参考集，输出两者之间的损失，所以有必要保证测试集中有5张图片作为参考集
        """
        writers = data_df['img_path'].apply(lambda x: x.split('/')[-2]).unique()
        train_writers, test_writers = train_test_split(writers, test_size=0.3, random_state=42)
        self.train_df = data_df[data_df['img_path'].apply(lambda x: x.split('/')[-2] in train_writers)]
        self.test_df = data_df[data_df['img_path'].apply(lambda x: x.split('/')[-2] in test_writers)]
        print(f'训练集: {len(self.train_df)} images (作者数: {len(train_writers)}) | 测试集: {len(self.test_df)} images (作者数: {len(test_writers)})')
        # self.train_df, self.test_df = train_test_split(data_df, test_size=0.3, shuffle=False, random_state=np.random.randint(0,100))
        print(f'训练集: {len(self.train_df)} images | 验证/测试集: {len(self.test_df)} images')

    # 魔法方法，这些方法通常由 Python 解释器在特定情况下自动调用，而不是由开发者直接调用
    def __len__(self):
        if self.mode == 'Train':
            return len(self.train_df)
        elif self.mode == 'Test':
            return len(self.test_df)
    
    # 论文中提到的reszie方法
    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        image = image.convert('L')
        image = np.asarray(image)
        # 计算图像的质量中心
        com = ndimage.measurements.center_of_mass(image)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        
        # 获取中心点的行和列
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :], image[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        # 从中心点向水平和垂直方向扫描，检测签名的起始和结束位置。
        for i, v in enumerate(c_col):
            v = np.sum(image[i, :])
            if v < 255*image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        # 同理，沿水平方向（X轴）扫描，找到签名的左右边界
        for j, v in enumerate(c_row):
            v = np.sum(image[:, j])
            if v < 255*image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        # 裁剪并返回 RGB 图像
        crop_rgb = Image.fromarray(np.asarray(image[x_start:x_end, y_start:y_end])).convert('RGB')
        return crop_rgb

    def __signature_to_binary__(self, image, target_size=None):
        """
        将签名图像转化为二值图（黑白图），并支持归一化
        参数:
            image: PIL.Image对象（原始签名图）
            target_size: 可选，目标尺寸元组（width, height）
        返回:
            np.ndarray: 二值化后的图像数组（0=背景, 255=签名）
        """
        # Step 1: 转换为灰度图
        if image.mode != 'L':
            image = image.convert('L')

        # Step 2: 归一化并反转（深色签名->255，浅色背景->0）
        img_array = np.array(image)
        normalized = 255 - ((img_array - img_array.min()) * (255 / (img_array.max() - img_array.min() or 1))).astype(np.uint8)

        # Step 3: 二值化（Otsu算法自动阈值）
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(normalized)
        binary_array = np.where(normalized >= threshold, 255, 0).astype(np.uint8)
    
        return binary_array

    def __getpatches__(self, x_arr):
        patches = []
        C, H, W = x_arr.shape # 3, 256, 256

        ### non-overlapping patches ###
        num_H = H // self.ptsz
        num_W = W // self.ptsz

        for i in range(num_H):
            for j in range(num_W):
                start_x = i*self.ptsz
                end_x = start_x + self.ptsz
                start_y = j*self.ptsz
                end_y = start_y + self.ptsz

                patch = x_arr[:, start_x:end_x, start_y:end_y]
                patch_tns = torch.from_numpy(patch)
                patches.append(torch.unsqueeze(patch_tns, 0))

        return torch.cat(patches, dim=0)

    def __getitem__(self, index):
        sample = {}
        if self.mode == 'Train':
            img_path = self.train_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            writer_id = img_path.split('/')[-2]
            label = self.train_df.iloc[index]['label']
            cropped_sig = self.__get_com_cropped__(sig_image)
            sig_image = self.basic_transforms(cropped_sig)
            sig_np = sig_image.numpy()
            sig_patches = self.__getpatches__(sig_np)

            sample = {'image' : sig_image, 'patches' : sig_patches, 'label' : label, 
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}

        elif self.mode == 'Test':
            img_path = self.test_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            writer_id = img_path.split('/')[-2]
            label = self.test_df.iloc[index]['label']
            cropped_sig = self.__get_com_cropped__(sig_image)
            sig_image = self.basic_transforms(cropped_sig)
            sig_np = sig_image.numpy()
            sig_patches = self.__getpatches__(sig_np)

            sample = {'image' : sig_image, 'patches' : sig_patches, 'label' : label,
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}
        
        return sample

def get_dataloader(args):
        train_dset = PreprocessAll(args, mode='Train')
        train_loader = DataLoader(train_dset, batch_size=args.batchsize, shuffle=True, num_workers=8)
        print('==> Train data loaded')
        
        test_dset = PreprocessAll(args, mode='Test')
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=8)
        print('==> Test data loaded')

        return train_loader, test_loader

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将归一化的图像张量反归一化"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 逆运算: t = (t * s) + m
    return tensor



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('PatchAttnReconstruction | SSL for Writer Identification')
    parser.add_argument('--base_dir', type=str, default='/home/admin-ps/Documents/hatAndMask/SignatureDatasets/')
    parser.add_argument('--dataset', type=str, default='ChiSig')
    parser.add_argument('--picPath', type=str, default='/home/admin-ps/Documents/hatAndMask/SignatureDatasets/ChiSig/白光启/白光启-20-1-true.jpg')
    parser.add_argument('--batchsize', type=int, default=1)  # change for Testing; default=16
    parser.add_argument('--print_freq', type=int,default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--Encoder_learning_rate', type=int, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--fmap_dims', type=int, default=8)
    parser.add_argument('--ptsz', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--is_pretrained', type=bool, default=False)
    parser.add_argument('--cls', type=str, default='knn')
    parser.add_argument('--model_path', type=str, default="/home/admin-ps/Documents/hatAndMask/signatureVerification/SURDS-SSL-OSV-main/PatchAttnRecons_SSL_OSV/saved_models/Chisig_ChiSig_R=None_SSL.pt")
    parser.add_argument('--stepsize', type=float, default=0.0005)
    args = parser.parse_args()

    print('\n'+'*'*100)
    train_loader, test_loader = get_dataloader(args)
    MODEL_PATH = args.model_path

    checkpoint = torch.load(MODEL_PATH)
    print(f"Loading model from: {MODEL_PATH} | Epochs trained: {checkpoint['epochs']}")
    # SSL类下的model，先创建一个实例
    model = SSL_Model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # 3. feature extraction from train and test
    features, labels, writer_ids, img_names = [], [], [], []
    mean_AP, mean_AN = 0.0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch['image'] = batch['image'].to(device)      # [N,3,256,256] 
            batch['patches'] = batch['patches'].to(device)  # [N,16,3,64,64]
            patch_attn_feats = []
            image_feature = model.encoder(batch['image'], pool=True)
            for patch in batch['patches'][0]: 
                patch_feature_map = model.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
                _, attn_feature = model.attn_module(image_feature, patch_feature_map) # [N,512]
                patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]
            patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
            patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
            patch_attn = model.avg_pool(patch_attn_feats) # [N,512,1]
            patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]
            features.append(patch_attn.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())
            writer_ids.append(batch['writer_id'][0])
            img_names.append(batch['img_name'][0])

    features = np.array(features)
    labels = np.array(labels)
    writer_ids = np.array(writer_ids)

    # print(features.shape, labels.shape)
    n_samples = features.shape[0] * features.shape[1]
    n_features = features.shape[2]

    features = features.reshape(n_samples, n_features)
    labels = labels.reshape(labels.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    print("使用KNN-Probe进行特征分类评估：")
    for k in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')  # 你也可以尝试 'euclidean'
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"K={k} | KNN-Probe 准确率: {acc:.4f}")

    tsne = TSNE(n_components=2, random_state=42)
    tsne_feats = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))
    plt.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.title("t-SNE of Signature Embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(label='Label (1=True, 0=Fake)')
    plt.grid(True)
    plt.savefig("tsne_features.png")


    # sample = processor.__getitem__()
    # # 在这里，他就已经是这种充满着黄色点的样本了
    # image = sample['image'].unsqueeze(0).to(device)
    # denorm_image = denormalize(sample['image'].clone())
    # save_image(denorm_image, "debug_image.png")
    # patches = sample['patches'].unsqueeze(0).to(device)
    # with torch.no_grad():
    #     image_feature = model.encoder(image, pool=True) # [N,512]
    #     patch_attn_feats = []
    #     patch_attn_maps = []
        
    #     for patch in patches[0]: 
    #     # patch = batch['patches'][0] # [64,3,32,32]
    #         patch_feature_map = model.encoder(patch.unsqueeze(0), pool=False) # [N,512,1,1]
    #         attn_map, attn_feature = model.attn_module(image_feature, patch_feature_map) # [N,512]
    #         patch_attn_feats.append(attn_feature.unsqueeze(1)) # [N,1,512]

    #     patch_attn_feats = torch.cat(patch_attn_feats, dim=1) # [N,16,512]
    #     patch_attn_feats = patch_attn_feats.permute(0,2,1) # [N,512,16]
    #     patch_attn = model.avg_pool(patch_attn_feats) # [N,512,1]
    #     patch_attn = patch_attn.permute(0,2,1).squeeze(1) # [N,512]

    #     recons_image = model.decoder(patch_attn) # [N,3,256,256]
    #     print(recons_image.shape)

    #     loss = F.mse_loss(recons_image, image)
    #     print(f"Reconstruction loss: {loss.item()}")
    #     pil_image = denormalize(recons_image.squeeze(0).cpu())
    #     save_image(pil_image, "reconsructed_image.png")
    #     print(f"重建图像已保存")

