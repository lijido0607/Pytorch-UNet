import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet, AttentionUNet
from utils.utils import plot_img_and_mask
def plot_prediction_with_ground_truth(img, pred_mask, true_mask=None):
    """
    显示2行2列的图像，包括原图、预测掩码、真实掩码和叠加效果
    
    参数:
        img: 原始图像
        pred_mask: 预测的掩码
        true_mask: 真实的掩码 (可选)
    """
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # 原始图像
    axs[0, 0].set_title('Original Image')
    axs[0, 0].imshow(img)
    axs[0, 0].axis('off')
    
    # 预测掩码
    axs[0, 1].set_title('Predicted Mask')
    axs[0, 1].imshow(pred_mask)
    axs[0, 1].axis('off')
    
    # 真实掩码 (如果提供)
    if true_mask is not None:
        axs[1, 0].set_title('Ground Truth Mask')
        axs[1, 0].imshow(true_mask)
    else:
        axs[1, 0].set_title('Ground Truth Mask (Not Available)')
        axs[1, 0].text(0.5, 0.5, 'N/A', 
                       horizontalalignment='center', 
                       verticalalignment='center',
                       transform=axs[1, 0].transAxes)
    axs[1, 0].axis('off')
    
    # 叠加效果
    axs[1, 1].set_title('Overlay')
    # 将预测掩码叠加在原图上
    img_array = np.array(img)
    mask_colored = np.zeros_like(img_array)
    if len(pred_mask.shape) == 2:  # 二值掩码
        mask_colored[pred_mask > 0] = [255, 0, 0]  # 红色标记预测区域
    plt.tight_layout()
    
    # 显示半透明叠加效果
    axs[1, 1].imshow(img)
    axs[1, 1].imshow(mask_colored, alpha=0.5)
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--attention', '-a', action='store_true', default=False, help='Use attention mechanism')
    parser.add_argument('--attention-type', type=str, default='cbam', 
                        choices=['cbam', 'channel', 'spatial', 'self_attention'],
                        help='Type of attention mechanism to use')
    parser.add_argument('--true-mask', '-tm', metavar='MASK', nargs='+', 
                       help='Filenames of ground truth masks (optional)')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    # Select model type based on args
    if args.attention:
        logging.info(f'Using Attention U-Net with {args.attention_type} attention mechanism')
        net = AttentionUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear, attention_type=args.attention_type)
    else:
        logging.info('Using standard U-Net model')
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            # logging.info(f'Visualizing results for image {filename}, close to continue...')
            # plot_img_and_mask(img, mask)
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            # 加载真实掩码（如果提供）
            true_mask = None
            if args.true_mask and i < len(args.true_mask):
                try:
                    true_mask_img = Image.open(args.true_mask[i])
                    true_mask = np.array(true_mask_img)
                except Exception as e:
                    logging.warning(f"Could not load ground truth mask: {e}")
            
            # 使用新的可视化函数
            plot_prediction_with_ground_truth(img, mask, true_mask)
