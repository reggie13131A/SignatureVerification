import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# np.random.seed(1)
# torch.manual_seed(1)

class Writer_Dataset(Dataset):
    # 要求传入一个args
    def __init__(self, args, mode):

        self.args = args
        self.mode = mode
        self.ptsz = args.ptsz

        # 这里做的是将resize、totensor、normalize整合到了一起
        self.basic_transforms = transforms.Compose([transforms.Resize((256,256)),
                                                    transforms.ToTensor(),
                                                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])
                                                    ])

        # 这里应该是数据增广
        self.augment_transforms = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.5),
                                                      transforms.RandomChoice([transforms.RandomEqualize(0.5),
                                                                               transforms.RandomInvert(0.5)]),
                                                      transforms.RandomApply([transforms.RandomAffine(
                                                          15, (0.05, 0.05), (1.5, 1.5), 0.5)], p=0.2)
                                                      ])

        # save all image paths with label (0/1)
        data_root = os.path.join(self.args.base_dir, self.args.dataset)  # BHSig260/Bengali
        print(f"目前的data_root是：{data_root}")

        # 这个变量是用于构建DataLoader的
        data_df = pd.DataFrame(columns=['img_path', 'label'])

        # for dir in os.listdir(data_root):
        #     print(dir)
        #     dir_path = os.path.join(data_root, dir)
        #     if os.path.isdir(dir_path):
        #         for img in os.listdir(dir_path):
        #             img_path = os.path.join(dir_path, img)
        #             img_split = img.split('-')
        #             """
        #                 通过解析文件名（例如，split('-') 分割文件名，检查第 4 个部分是否为 G），自动生成标签。
        #                 这种方法依赖于数据集文件名的标准格式。
        #             """
        #             label = 1 if img_split[3] == 'G' else 0
        #             data_df = data_df.append({'img_path': img_path, 'label': label}, ignore_index=True)

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
        # x = Image.fromarray(x).convert('RGB').resize((256,256))
        # x_arr = np.asarray(x)
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
                # print(patch.shape)
                patch_tns = torch.from_numpy(patch)
                patches.append(torch.unsqueeze(patch_tns, 0))

        ### 50% pixel overlapping ###
        # num_H = (H//(self.ptsz//2)) - 1
        # num_W = (W//(self.ptsz//2)) - 1

        # for i in range(num_H):
        #     for j in range(num_W):
        #         start_x = i*(self.ptsz//2)
        #         end_x = start_x + self.ptsz
        #         start_y = j*(self.ptsz//2)
        #         end_y = start_y + self.ptsz

        #         patch = x_arr[:, start_x:end_x, start_y:end_y]
        #         # print(patch.shape)
        #         patch_tns = torch.from_numpy(patch)
        #         patches.append(torch.unsqueeze(patch_tns, 0))

        return torch.cat(patches, dim=0)

    def __getitem__(self, index):
        sample = {}
        if self.mode == 'Train':
            img_path = self.train_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            # 这种是根据某一个数据集来进行diy的，得改
            writer_id = img_path.split('/')[-2]
            label = self.train_df.iloc[index]['label']
            # 在这里加上了二值化
            # bin_sig = self.__signature_to_binary__(sig_image)
            cropped_sig = self.__get_com_cropped__(sig_image)
            # sig_image = self.basic_transforms(self.augment_transforms(cropped_sig))
            sig_image = self.basic_transforms(cropped_sig)
            sig_np = sig_image.numpy()
            # print("signp has shape: "+ str(sig_np.shape))
            sig_patches = self.__getpatches__(sig_np)

            sample = {'image' : sig_image, 'patches' : sig_patches, 'label' : label, 
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}

        elif self.mode == 'Test':
            img_path = self.test_df.iloc[index]['img_path']
            sig_image = Image.open(img_path)
            # 这里也得改
            writer_id = img_path.split('/')[-2]
            label = self.test_df.iloc[index]['label']
            # bin_sig = self.__signature_to_binary__(sig_image)
            cropped_sig = self.__get_com_cropped__(sig_image)
            sig_image = self.basic_transforms(cropped_sig)
            sig_np = sig_image.numpy()
            # print("signp has shape: "+ str(sig_np.shape))
            sig_patches = self.__getpatches__(sig_np)

            sample = {'image' : sig_image, 'patches' : sig_patches, 'label' : label,
                      'writer_id' : writer_id, 'img_name' : os.path.basename(img_path)}
        
        return sample


def get_dataloader(args):
        train_dset = Writer_Dataset(args, mode='Train')
        train_loader = DataLoader(train_dset, batch_size=args.batchsize, shuffle=True, num_workers=8)
        print('==> Train data loaded')
        
        test_dset = Writer_Dataset(args, mode='Test')
        # 使用dataLoader类来创建实例，会自动调用 Writer_Dataset 的 __getitem__ 方法
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=8)
        print('==> Test data loaded')

        return train_loader, test_loader
