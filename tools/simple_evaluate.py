#!/usr/bin/env python3
"""
简单的SAM2模型评估脚本
计算预测掩码与真实掩码的IoU和Dice指标
"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from collections import defaultdict


def load_mask(mask_path):
    """加载PNG掩码文件"""
    if not os.path.exists(mask_path):
        return None
    mask = np.array(Image.open(mask_path)).astype(np.uint8)
    return mask


def get_unique_objects(mask):
    """获取掩码中的唯一对象ID（排除背景0）"""
    unique_ids = np.unique(mask)
    return unique_ids[unique_ids > 0].tolist()


def compute_iou(pred_mask, gt_mask, obj_id):
    """计算单个对象的IoU"""
    pred_obj = (pred_mask == obj_id)
    gt_obj = (gt_mask == obj_id)
    
    intersection = np.logical_and(pred_obj, gt_obj).sum()
    union = np.logical_or(pred_obj, gt_obj).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(pred_mask, gt_mask, obj_id):
    """计算单个对象的Dice系数"""
    pred_obj = (pred_mask == obj_id)
    gt_obj = (gt_mask == obj_id)
    
    intersection = np.logical_and(pred_obj, gt_obj).sum()
    total = pred_obj.sum() + gt_obj.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / total


def evaluate_predictions(pred_dir, gt_dir, file_list_path, output_path=None):
    """评估预测结果"""
    
    # 读取文件列表
    with open(file_list_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    
    metrics = defaultdict(list)
    successful_samples = 0
    
    print(f"Evaluating {len(file_names)} samples...")
    
    for file_name in tqdm(file_names):
        # 构建文件路径
        pred_path = os.path.join(pred_dir, file_name, '1.png')  # 预测掩码
        gt_path = os.path.join(gt_dir, file_name, '1.png')      # 真实掩码
        
        # 检查文件是否存在
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction not found for {file_name}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {file_name}")
            continue
        
        # 加载掩码
        pred_mask = load_mask(pred_path)
        gt_mask = load_mask(gt_path)
        
        if pred_mask is None or gt_mask is None:
            continue
        
        # 确保掩码尺寸一致
        if pred_mask.shape != gt_mask.shape:
            print(f"Warning: Shape mismatch for {file_name}: {pred_mask.shape} vs {gt_mask.shape}")
            continue
        
        # 获取真实掩码中的对象ID
        gt_obj_ids = get_unique_objects(gt_mask)
        if not gt_obj_ids:
            continue
        
        # 计算每个对象的指标
        sample_ious = []
        sample_dices = []
        
        for obj_id in gt_obj_ids:
            iou = compute_iou(pred_mask, gt_mask, obj_id)
            dice = compute_dice(pred_mask, gt_mask, obj_id)
            
            sample_ious.append(iou)
            sample_dices.append(dice)
            
            metrics[f'iou_obj_{obj_id}'].append(iou)
            metrics[f'dice_obj_{obj_id}'].append(dice)
        
        # 样本级别的平均指标
        if sample_ious:
            metrics['iou_per_sample'].append(np.mean(sample_ious))
            metrics['dice_per_sample'].append(np.mean(sample_dices))
            successful_samples += 1
    
    if successful_samples == 0:
        print("No valid samples found for evaluation!")
        return
    
    # 计算总体统计
    mean_iou = np.mean(metrics['iou_per_sample'])
    std_iou = np.std(metrics['iou_per_sample'])
    mean_dice = np.mean(metrics['dice_per_sample'])
    std_dice = np.std(metrics['dice_per_sample'])
    
    # 输出结果
    print(f"\n=== Evaluation Results ===")
    print(f"Successfully evaluated: {successful_samples}/{len(file_names)} samples")
    print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    
    # 保存详细结果
    results = {
        'total_files': len(file_names),
        'successful_samples': successful_samples,
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'mean_dice': float(mean_dice),
        'std_dice': float(std_dice),
        'per_sample_iou': [float(x) for x in metrics['iou_per_sample']],
        'per_sample_dice': [float(x) for x in metrics['dice_per_sample']]
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 prediction results')
    parser.add_argument('--pred-dir', required=True, 
                      help='Directory containing predicted masks (output from vos_inference.py)')
    parser.add_argument('--gt-dir', required=True,
                      help='Directory containing ground truth masks')
    parser.add_argument('--file-list', required=True,
                      help='Text file containing list of sample names to evaluate')
    parser.add_argument('--output', default='evaluation_results.json',
                      help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_predictions(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        file_list_path=args.file_list,
        output_path=args.output
    )


if __name__ == '__main__':
    main()