import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # 新增变量
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    # iterate over the validation set
    
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                # Ensure mask_true has shape (B, C, H, W)
                if mask_true.ndim == 3:
                    mask_true = mask_true.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

                if mask_pred.shape != mask_true.shape:
                    mask_pred = F.interpolate(mask_pred, size=mask_true.shape[2:], mode='bilinear', align_corners=False)
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                                # 计算TP, FP, TN, FN
                tp = torch.sum((mask_pred == 1) & (mask_true == 1)).item()
                fp = torch.sum((mask_pred == 1) & (mask_true == 0)).item()
                tn = torch.sum((mask_pred == 0) & (mask_true == 0)).item()
                fn = torch.sum((mask_pred == 0) & (mask_true == 1)).item()
                
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                # Ensure mask_true has shape (B, C, H, W)
                if mask_true.ndim == 3:
                    mask_true = mask_true.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                if mask_pred.shape != mask_true.shape:
                    mask_pred = F.interpolate(mask_pred, size=mask_true.shape[2:], mode='bilinear', align_corners=False)
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
    # 计算指标
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # # 打印metrics
    # print(f'Validation metrics:')
    # print(f'Dice score: {dice_score / max(num_val_batches, 1):.4f}')
    # print(f'Accuracy: {accuracy:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'Precision: {precision:.4f}')
    # print(f'F1 Score: {f1:.4f}')

    net.train()
    # 返回字典包含所有指标
    return {
        'dice': dice_score / max(num_val_batches, 1),
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
    # net.train()
    # return dice_score / max(num_val_batches, 1)
