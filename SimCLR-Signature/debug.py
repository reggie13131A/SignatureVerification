import os

txt_path = "/home/Work/signatureVerification/Research/SimCLR-master/invalid_images.txt"

with open(txt_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

img_paths = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # 去掉 "图片已经损坏：" 或 "全像素相同（空白图）:" 前缀
    if ":" in line:
        line = line.split(":", 1)[1]
    img_paths.append(line)

deleted_count = 0
for path in img_paths:
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"已删除: {path}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {path}: {e}")
    else:
        print(f"文件不存在: {path}")

print(f"\n总计删除 {deleted_count} 张坏图。")