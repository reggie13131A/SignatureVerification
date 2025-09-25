import torchvision.transforms as T
import torch
from PIL import Image
import os
import numpy as np
import random
import cv2


class RandomAffineStretchTranslate:
    """
    AVN风格几何不变性增强：随机横纵拉伸 + 平移，再resize回原尺寸
    """
    def __init__(self, scale_x_range=(0.9, 1.1), scale_y_range=(0.9, 1.1),
                 translate_range=(0.05, 0.05), p=0.8, output_size=(256, 256)):
        self.scale_x_range = scale_x_range
        self.scale_y_range = scale_y_range
        self.translate_range = translate_range  # (max_dx_ratio, max_dy_ratio)
        self.p = p
        self.output_size = output_size

    def __call__(self, img):
        if random.random() > self.p:
            return img.resize(self.output_size, Image.BICUBIC)

        w, h = img.size

        # 随机采样横纵比例缩放
        scale_x = random.uniform(*self.scale_x_range)
        scale_y = random.uniform(*self.scale_y_range)
        new_w = max(1, int(w * scale_x))
        new_h = max(1, int(h * scale_y))

        # 缩放
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # 随机平移（通过创建大画布再粘贴实现）
        max_dx = int(self.translate_range[0] * w)
        max_dy = int(self.translate_range[1] * h)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        canvas = Image.new("RGB", (w, h), (255, 255, 255))  # 白底画布
        paste_x = min(max(0, dx), w - new_w)
        paste_y = min(max(0, dy), h - new_h)
        canvas.paste(img, (paste_x, paste_y))

        # 最终resize到指定大小
        return canvas.resize(self.output_size, Image.BICUBIC)


class RandomNonUniformBrightness:
    """
    随机生成非均匀亮度扰动，模拟 AVN 论文中的变换不变性增强
    """
    def __init__(self, scale_range=(0.7, 1.3), p=0.8):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img  # 不执行扰动

        # 转成 numpy
        img_np = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]

        H, W = img_np.shape[:2]

        # 随机生成亮度 mask
        mask = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(H, W)).astype(np.float32)

        # 高斯平滑 mask，避免硬边界
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=W/10, sigmaY=H/10, borderType=cv2.BORDER_REFLECT)

        # 扩展到 RGB
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)  # 扩展到 3 通道

        # 应用 mask
        img_aug = img_np * mask
        img_aug = np.clip(img_aug, 0, 1)

        # 转回 PIL
        img_aug = (img_aug * 255).astype(np.uint8)
        return Image.fromarray(img_aug)

# 弱增广流水线
weak_aug = T.Compose([
    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(0.4, 0.4, 0.4, 0.2),
    T.Resize((256, 256)),
    T.ToTensor()
])

from data_aug.gaussian_blur import GaussianBlur
strong_aug = T.Compose([
    T.RandomRotation(degrees=(-7, 7)),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.RandomApply([GaussianBlur(kernel_size=3)], p=0.4),  # 替代 ElasticTransformation
    T.RandomApply([T.RandomErasing(p=0.3, scale=(0.02, 0.1))], p=0.3),        # 替代 Morphology
    T.Resize((256, 256)),
    T.ToTensor()
])

# 组合增广流水线
compose = T.Compose([
    # T.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    RandomNonUniformBrightness(scale_range=(0.7, 1.3), p=1.0),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
    T.RandomApply([T.RandomErasing(p=0.3, scale=(0.02, 0.1))], p=0.3),
    T.Resize((256, 256)),
    T.ToTensor()
])

# AVN 方式下的数据增强
AVN_aug = T.Compose([
    RandomAffineStretchTranslate(scale_x_range=(0.9, 1.1),
                                  scale_y_range=(0.9, 1.1),
                                  translate_range=(0.05, 0.05),
                                  p=1.0,  # 100% 执行
                                  output_size=(256, 256)),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.ToTensor()
])

# 加载签名对
def load_signature_pairs(signature1, signature2):
    pos_sample1 = strong_aug(signature1)  # 正样本：强增广
    pos_sample2 = weak_aug(signature1)    # 正样本：弱增广
    pos_sample3 = compose(signature1)     # 组合增广
    pos_sample4 = AVN_aug(signature1)     # 组合增广
    neg_sample = weak_aug(signature2)     # 负样本：弱增广
    return pos_sample1, pos_sample2, pos_sample3, pos_sample4, neg_sample

# 张量转图片并保存
def save_tensor_as_image(tensor, output_path):
    from torchvision.transforms import ToPILImage
    image = ToPILImage()(tensor)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

# 主程序
if __name__ == "__main__":
    # 加载签名图像，确保转换为 RGB
    img = Image.open("/home/Work/signatureDatasets/processed/艾飞艳.png").convert('RGB')
    forg = Image.open("/home/Work/signatureDatasets/processed/艾芬.png").convert('RGB')

    # 获取增广后的张量
    pos_sample1, pos_sample2, pos_sample3, pos_sample4, neg_sample = load_signature_pairs(img, forg)

    # 保存增广后的图像
    save_tensor_as_image(pos_sample1, "./augmented_images/pos_sample1_strong.jpg")
    save_tensor_as_image(pos_sample2, "./augmented_images/pos_sample2_weak.jpg")
    save_tensor_as_image(pos_sample3, "./augmented_images/pos_sample3_compose.jpg")
    save_tensor_as_image(pos_sample4, "./augmented_images/pos_sample4_compose.jpg")
    save_tensor_as_image(neg_sample, "./augmented_images/neg_sample_weak.jpg")

    print("增广后的图像已保存至 ./augmented_images/ 目录")


