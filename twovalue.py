import os
from PIL import Image
import numpy as np

mask_dir = "data/masks_binary" # 原始掩膜目录
output_dir = "data/masks"    # 新的掩膜目录
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(mask_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path = os.path.join(mask_dir, filename)
        img = Image.open(path).convert("L")
        img_np = np.array(img)

        # 将像素值归一化为 0 或 1（大于128算前景）
        binary_np = (img_np > 127).astype(np.uint8)

        # 保存为 PNG（乘 255 保证显示正常）
        binary_img = Image.fromarray(binary_np * 255)
        binary_img.save(os.path.join(output_dir, filename))

        print(f"✅ 处理完成: {filename}")


