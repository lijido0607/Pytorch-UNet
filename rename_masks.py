import os
from PIL import Image

img_dir = "data/imgs"
mask_dir = "data/masks"

# 遍历图像文件夹，找到图像 ID（不含扩展名）
image_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_id in image_ids:
    original_mask_path = os.path.join(mask_dir, f"{img_id}.jpg")
    new_mask_path = os.path.join(mask_dir, f"{img_id}_mask.png")

    if os.path.exists(original_mask_path):
        img = Image.open(original_mask_path)
        img.save(new_mask_path)
        print(f"✅ 已保存：{new_mask_path}")
    else:
        print(f"⚠️ 没找到掩膜：{original_mask_path}")
