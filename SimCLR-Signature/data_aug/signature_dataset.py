import os
import pandas as pd
import numpy as np
from PIL import Image
from scipy import ndimage
import random
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.debug import RandomAffineStretchTranslate, RandomNonUniformBrightness


class SignatureContrastiveLearningDataset:
    def __init__(self, root_folder, debug_aug=False, debug_samples=5):
        self.root_folder = root_folder
        self.debug_aug = debug_aug
        self.debug_samples = debug_samples

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """签名友好的 SimCLR 增强管道"""
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.1 * s)
        data_transforms = transforms.Compose([
            transforms.RandomApply([color_jitter], p=0.8),
            RandomNonUniformBrightness(scale_range=(0.85, 1.15), p=0.5),
            transforms.RandomGrayscale(p=0.2),
            RandomAffineStretchTranslate(scale_x_range=(0.85, 1.15),
                                         scale_y_range=(0.85, 1.15),
                                         translate_range=(0.05, 0.05),
                                         p=1.0,
                                         output_size=(size, size)),
            transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * size))], p=0.4),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return data_transforms

    def _get_com_cropped(self, image, img_path=None):
        image = image.convert('L')
        image_arr = np.asarray(image)

        # 提前检测空白图
        if np.all(image_arr == image_arr[0, 0]):
            with open("invalid_images.txt", "a", encoding="utf-8") as f:
                f.write(f"全像素相同（空白图）:{img_path}\n")
            return image.convert('RGB')

        com = ndimage.center_of_mass(image_arr)

        if np.isnan(com[0]) or np.isnan(com[1]):
            with open("invalid_images.txt", "a", encoding="utf-8") as f:
                f.write(f"质心计算失败:{img_path}\n")
            return image.convert('RGB')

        com = np.round(com)
        com[0] = np.clip(com[0], 0, image_arr.shape[0])
        com[1] = np.clip(com[1], 0, image_arr.shape[1])

        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image_arr[X_center, :], image_arr[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1
        for i, v in enumerate(c_col):
            if np.sum(image_arr[i, :]) < 255 * image_arr.shape[1]:
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            if np.sum(image_arr[:, j]) < 255 * image_arr.shape[0]:
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        if x_start == -1 or x_end == -1 or y_start == -1 or y_end == -1:
            with open("invalid_images.txt", "a", encoding="utf-8") as f:
                f.write(f"裁剪范围异常: {img_path}\n")
            return image.convert('RGB')

        crop_rgb = Image.fromarray(image_arr[x_start:x_end, y_start:y_end]).convert('RGB')
        return crop_rgb


    def get_dataset(self, name, n_views):
        """根据数据集名称返回 dataset 对象"""
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(32),
                                                    n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                            transform=ContrastiveLearningViewGenerator(
                                                self.get_simclr_pipeline_transform(96),
                                                n_views),
                                            download=True),

            'signature': lambda: self._load_signature_dataset(
                os.path.join(self.root_folder, 'processed'),
                n_views
            )
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def _load_signature_dataset(self, data_path, n_views):
        """加载签名数据集"""
        all_images = []
        for root, _, files in os.walk(data_path):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    all_images.append(os.path.join(root, fname))

        df = pd.DataFrame({'img_path': all_images})
        print(f"共加载 {len(df)} 张签名图片")

        dataset_self = self
        simclr_transform = self.get_simclr_pipeline_transform(256)

        class SignatureDataset(Dataset):
            def __len__(self):
                return len(df)

            def __getitem__(self, idx):
                img_path = df.iloc[idx]['img_path']
                img = Image.open(img_path)
                cropped = dataset_self._get_com_cropped(img, img_path)
                view1, view2 = ContrastiveLearningViewGenerator(simclr_transform, n_views=2)(cropped)
                return view1, view2

        sig_dataset = SignatureDataset()

        # ===== 调试模式：保存增强对比 =====
        if self.debug_aug:
            import torchvision.utils as vutils
            save_dir = "/home/Work/signatureVerification/Research/SimCLR-master/data_aug/augmented_images"
            os.makedirs(save_dir, exist_ok=True)

            resize_to_256 = transforms.Resize((256, 256))

            for _ in range(min(self.debug_samples, len(sig_dataset))):
                idx = random.randint(0, len(sig_dataset) - 1)
                img_path = df.iloc[idx]['img_path']
                img = Image.open(img_path)
                cropped = dataset_self._get_com_cropped(img)
                cropped_resized = resize_to_256(cropped)  # 保证尺寸一致

                views = ContrastiveLearningViewGenerator(simclr_transform, n_views=2)(cropped_resized)

                # 拼图保存
                grid = vutils.make_grid(
                    [transforms.ToTensor()(cropped_resized)] + [v for v in views],
                    nrow=3, padding=2, normalize=True
                )
                save_path = os.path.join(save_dir, f"sample_{os.path.basename(img_path)}.png")
                vutils.save_image(grid, save_path)

            print(f"[Debug] 已保存 {self.debug_samples} 张增强对比图到 {save_dir}")

        return sig_dataset
