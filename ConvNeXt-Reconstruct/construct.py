import os
from PIL import Image
import numpy as np

def rotate_left_90(image: np.ndarray) -> np.ndarray:
    """
    Rotate image 90 degrees counter-clockwise (left).
    """
    return np.rot90(image, k=1)

def remove_transparency(image_path, background_color=(255, 255, 255)) -> Image.Image:
    """
    Remove transparency from image and fill with background_color.
    """
    img = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", img.size, background_color + (255,))
    composite = Image.alpha_composite(background, img)
    return composite.convert("RGB")

def process_folder(input_dir, output_dir):
    supported_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    count = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_ext:
                input_path = os.path.join(root, fname)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    # Step 1: Remove transparency (returns PIL image)
                    image = remove_transparency(input_path)

                    # Step 2: Rotate (need to convert to np.array first)
                    rotated = rotate_left_90(np.array(image))

                    # Step 3: Convert back to PIL and save
                    result = Image.fromarray(rotated)
                    result.save(output_path)

                    print(f"[✔] Processed: {input_path} -> {output_path}")
                    count += 1
                except Exception as e:
                    print(f"[✘] Failed: {input_path} | Reason: {e}")

    print(f"\n✅ Finished processing {count} images.")

# 示例用法
if __name__ == "__main__":
    input_folder = "/home/Work/signatureDatasets/PreTrain"       # 替换为你的输入路径
    output_folder = "/home/Work/signatureDatasets/processed"  # 替换为输出路径
    process_folder(input_folder, output_folder)

