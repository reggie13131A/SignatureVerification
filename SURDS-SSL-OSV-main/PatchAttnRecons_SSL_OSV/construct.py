"""
    这个文件里面的所有函数是用来构造数据集的；
    warning: 这个文件下的所有函数会对文件名字后缀进行更改；
    并且更改的文件名后缀是-true，代表着真实的签名；
    这个文件的更改内容是适配于ChiSig数据集的；如果有其他的数据集，请自定义
"""


import os
import random
from collections import defaultdict
import pandas as pd

def process_signature_folder(folder_path):
    """处理单个签名文件夹"""
    if not os.path.exists(folder_path):
        return
    
    # 获取所有原始文件（不包含已经添加-true的文件）
    files_with_ids = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') and '-true' not in filename:
            parts = filename.split('-')
            if len(parts) >= 2:
                try:
                    file_id = int(parts[1])
                    files_with_ids.append((filename, file_id))
                except ValueError:
                    continue
    
    if not files_with_ids:
        return
    
    # 按ID分组所有文件
    id_groups = defaultdict(list)
    for filename, file_id in files_with_ids:
        id_groups[file_id].append(filename)
    
    # 检查是否存在满足id1 - id2 = 100的配对
    files_to_rename = []
    ids = list(id_groups.keys())
    
    # 寻找满足差值为100的配对
    found_pairs = False
    for id1 in ids:
        id2 = id1 - 100
        if id2 in id_groups:  # 直接检查是否存在id2 = id1 - 100
            # 只处理id2组的文件
            for filename in id_groups[id2]:
                files_to_rename.append(filename)
            found_pairs = True
            break  # 找到一个配对即可
    
    # 如果没找到满足条件的配对，随机选择一个ID下的所有文件
    if not found_pairs:
        print(f"文件夹 {os.path.basename(folder_path)} 未找到满足条件的配对，随机选择一组文件标记为true")
        # 随机选择一个ID
        random_id = random.choice(ids)
        # 选择该ID下的所有文件
        for filename in id_groups[random_id]:
            files_to_rename.append(filename)
    
    # 执行重命名操作
    for filename in files_to_rename:
        old_path = os.path.join(folder_path, filename)
        
        # 检查文件是否存在
        if not os.path.exists(old_path):
            print(f"文件不存在，跳过: {filename}")
            continue
            
        # 构造新文件名：在扩展名前插入"-true"
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            new_filename = name_parts[0] + "-true." + name_parts[1]
        else:
            new_filename = filename + "-true"
        
        new_path = os.path.join(folder_path, new_filename)
        
        try:
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")
        except FileNotFoundError:
            print(f"文件已被处理或不存在: {filename}")
        except Exception as e:
            print(f"重命名失败: {filename}, 错误: {e}")

def process_all_folders(data_root):
    """处理data_root下的所有文件夹"""
    if not os.path.exists(data_root):
        print(f"路径不存在: {data_root}")
        return
    
    # 遍历所有子文件夹
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            print(f"\n处理文件夹: {item}")
            process_signature_folder(item_path)




def remove_true_from_filename(filepath):
    """从文件名中移除-true字符串"""
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # 检查文件名是否包含-true
    if '-true' in filename:
        # 构造新文件名（移除-true）
        new_filename = filename.replace('-true', '')
        new_filepath = os.path.join(dirname, new_filename)
        
        # 避免重复处理
        if new_filepath != filepath:
            try:
                os.rename(filepath, new_filepath)
                print(f"重命名: {filename} -> {new_filename}")
            except Exception as e:
                print(f"重命名失败: {filename}, 错误: {e}")

def process_folder(folder_path):
    """处理文件夹中的所有文件"""
    if not os.path.exists(folder_path):
        print(f"路径不存在: {folder_path}")
        return
    
    # 遍历所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 只处理.jpg文件（可根据需要修改）
            if filename.endswith('.jpg'):
                filepath = os.path.join(root, filename)
                remove_true_from_filename(filepath)

def main():
    # 设置要处理的根文件夹路径
    data_root = "/home/admin-ps/Documents/hatAndMask/SignatureDatasets/ChiSig"
    
    print("开始处理文件夹...")
    # process_folder(data_root)
    process_all_folders(data_root)
    print("处理完成！")


# 假设这是您的DataLoader类的一部分
class SignatureDataLoader:
    def __init__(self, base_dir, dataset):
        self.base_dir = base_dir
        self.dataset = dataset
    
    def prepare_data(self):
        data_root = os.path.join(self.base_dir, self.dataset)  # BHSig260/Bengali
        print(f"目前的data_root是：{data_root}")

        # 创建DataFrame用于构建DataLoader
        data_df = pd.DataFrame(columns=['img_path', 'label'])

        for dir_name in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir_name)
            if os.path.isdir(dir_path):
                for img_name in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_name)
                    
                    # 新的标签判断逻辑：文件名中包含"-true"则标记为1，否则为0
                    label = 1 if '-true' in img_name else 0
                    
                    data_df = data_df._append({'img_path': img_path, 'label': label}, ignore_index=True)

        print(f'{self.dataset} 包含总计 {len(data_df)} 张图片!!')

data_root = "/home/admin-ps/Documents/hatAndMask/SignatureDatasets"
dataset = "ChiSig"
dataloader = SignatureDataLoader(data_root, dataset)
dataloader.prepare_data()