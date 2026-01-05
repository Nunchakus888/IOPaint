#!/usr/bin/env python3
"""
水印去除流水线：
1. 按尺寸分组生成 masks (复用 gemini2.py)
2. 使用 iopaint + lama 模型批量去除水印
3. （可选）使用 MAT 模型二次修复残留
4. （可选）使用 RealESRGAN 插件增强图片质量
"""
import os

from config import (
    INPUT_DIR, MASK_DIR, OUTPUT_DIR,
    INPAINT_MODEL, DEVICE, ENABLE_ENHANCE, ENABLE_REFINE, VERBOSE, MAX_WORKERS
)
from gemini2 import scan_images_recursive, extract_watermarks_by_group
from watermark_remover import batch_remove_watermarks, load_masks, refine_with_mat, get_image_size
from enhancer import ImageEnhancer


def run_pipeline():
    """执行完整流水线"""
    
    # Step 1: 生成 Masks
    print("=" * 50)
    print("📌 Step 1: 生成分尺寸 Masks")
    print("=" * 50)
    # extract_watermarks_by_group(INPUT_DIR, MASK_DIR)
    
    # Step 2: 去水印
    features = []
    if ENABLE_REFINE:
        features.append("MAT二次修复")
    if ENABLE_ENHANCE:
        features.append("RealESRGAN增强")
    
    step_desc = " + ".join(features) if features else ""
    print("=" * 50)
    print(f"📌 Step 2: 批量去除水印" + (f" ({step_desc})" if step_desc else ""))
    print("=" * 50)
    
    images = scan_images_recursive(INPUT_DIR)
    masks = load_masks(MASK_DIR)
    print(f"\n🖼️  待处理图片: {len(images)} 张")
    print(f"🎭 可用 Masks: {list(masks.keys())}")
    print(f"⚡ 并发数: {MAX_WORKERS}\n")
    
    # 初始化增强器（懒加载）
    enhancer = ImageEnhancer(DEVICE) if ENABLE_ENHANCE else None
    
    def on_progress(img_path, output_path):
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        print(f"🔧 去水印: {rel_path}")
        
        # 二次修复（用 MAT 处理残留）
        if ENABLE_REFINE and os.path.exists(output_path):
            size = get_image_size(img_path)
            mask_name = f"mask_{size[0]}x{size[1]}.png"
            if mask_name in masks:
                print(f"🔄 MAT修复: {rel_path}")
                refine_with_mat(output_path, masks[mask_name], output_path, DEVICE, verbose=VERBOSE)
        
        # RealESRGAN 增强
        if enhancer and os.path.exists(output_path):
            print(f"✨ 增强中: {rel_path}")
            enhancer.enhance(output_path)
    
    results = batch_remove_watermarks(
        images=images,
        input_dir=INPUT_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        model=INPAINT_MODEL,
        device=DEVICE,
        on_progress=on_progress,
        max_workers=MAX_WORKERS
    )
    
    print(f"\n✅ 完成！处理 {len(results)} 张图片")
    print(f"📁 输出目录: {OUTPUT_DIR}")


def main():
    # 切换到 v2 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    run_pipeline()


if __name__ == '__main__':
    main()
