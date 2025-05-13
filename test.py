import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Step 1: 加载模型结构
from unet import UNet  # 你自己的模型文件和类名

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1)  # 根据你的模型参数调整
model.load_state_dict(torch.load('checkpoints\\checkpoint_epoch1.pth', map_location=device))
model.to(device)
model.eval()

# Step 2: 加载图像并预处理
image_path = 'data\\imgs\\0313-1_240.jpg'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 你训练时的尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 替换为训练时使用的 mean/std
])

input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

# Step 3: 模型推理
with torch.no_grad():
    output = model(input_tensor)  # [1, 1, H, W]
    pred_mask = torch.sigmoid(output)
    binary_mask = (pred_mask > 0.5).float().squeeze().cpu().numpy()  # [H, W]

# Step 4: 可视化
def visualize_result(image_pil, mask_np):
    image_np = np.array(image_pil.resize((256, 256)))  # [H, W, 3]
    mask_colored = np.zeros_like(image_np)
    mask_colored[mask_np > 0.5] = [255, 0, 0]  # 红色区域

    overlay = image_np.copy()
    alpha = 0.5
    overlay[mask_np > 0.5] = (1 - alpha) * image_np[mask_np > 0.5] + alpha * mask_colored[mask_np > 0.5]

    # 可视化三图
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_pil)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Predicted Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay.astype(np.uint8))
    plt.title("Overlay")

    plt.tight_layout()
    plt.show()

# Run it
visualize_result(image, binary_mask)
