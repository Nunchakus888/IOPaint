#!/usr/bin/env python3
"""
水印去除流水线：
1. 按尺寸分组生成 masks (复用 gemini2.py)
2. 使用 iopaint + lama 模型批量去除水印
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import subprocess
import cv2
from gemini2 import scan_images_recursive, extract_watermarks_by_group

# === 配置 ===
INPUT_DIR = 'enhance_analysis/images'
MASK_DIR = 'enhance_analysis/masks'
OUTPUT_DIR = 'enhance_analysis/output'
MODEL = 'lama'
DEVICE = 'cpu'


def get_image_size(path):
    """获取图片尺寸 (w, h)"""
    img = cv2.imread(path)
    return (img.shape[1], img.shape[0]) if img is not None else None


def remove_watermarks(input_dir, mask_dir, output_dir):
    """根据尺寸匹配 mask，批量去除水印"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 扫描所有待处理图片
    images = scan_images_recursive(input_dir)
    print(f"\n🖼️  待处理图片: {len(images)} 张\n")
    
    # 加载可用的 masks
    masks = {f: os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')}
    print(f"🎭 可用 Masks: {list(masks.keys())}\n")
    
    for img_path in images:
        size = get_image_size(img_path)
        if not size:
            continue
        
        mask_name = f"mask_{size[0]}x{size[1]}.png"
        if mask_name not in masks:
            print(f"⚠️  跳过 (无匹配mask): {os.path.basename(img_path)}")
            continue
        
        # 构建输出路径，保持子目录结构
        rel_path = os.path.relpath(img_path, input_dir)
        out_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(out_subdir, exist_ok=True)
        
        # 调用 iopaint
        cmd = [
            'iopaint', 'run',
            f'--model={MODEL}',
            f'--device={DEVICE}',
            f'--image={img_path}',
            f'--mask={masks[mask_name]}',
            f'--output={out_subdir}'
        ]
        
        print(f"🔧 处理: {rel_path}")
        subprocess.run(cmd, capture_output=True)
    
    print(f"\n✅ 完成！输出目录: {output_dir}")


def main():
    # 切换到 v2 目录（enhance_analysis 的父目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    print("=" * 50)
    print("📌 Step 1: 生成分尺寸 Masks")
    print("=" * 50)
    extract_watermarks_by_group(INPUT_DIR, MASK_DIR)
    
    print("=" * 50)
    print("📌 Step 2: 批量去除水印")
    print("=" * 50)
    remove_watermarks(INPUT_DIR, MASK_DIR, OUTPUT_DIR)


if __name__ == '__main__':
    main()

