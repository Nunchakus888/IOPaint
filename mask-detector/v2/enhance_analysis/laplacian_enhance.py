"""
Laplacian增强器 - 快速生成边缘增强图像
=============================================

原理：
- Laplacian算子是二阶导数算子
- 对边缘响应非常敏感
- 可以检测快速变化的灰度区域（文字边缘）

为什么对水印有效：
- 文字有锐利的边缘
- 即使对比度很低，边缘的二阶导数仍不为零
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def laplacian_enhance(image: np.ndarray, return_binary: bool = True) -> np.ndarray:
    """
    Laplacian增强
    
    Args:
        image: BGR格式输入图像
        return_binary: 是否返回二值化结果，False则返回归一化灰度图
        
    Returns:
        增强后的图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 轻微模糊去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Laplacian算子 (ksize=3: 3x3的Laplacian核)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    
    # 取绝对值
    laplacian_abs = np.abs(laplacian)
    
    # 归一化到0-255
    lap_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if not return_binary:
        return lap_norm
    
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(
        lap_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, -2
    )
    
    return thresh


def process_single_image(image_path: str, output_path: str, return_binary: bool = True):
    """处理单张图片"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load: {image_path}")
        return False
    
    # 生成Laplacian增强图
    result = laplacian_enhance(image, return_binary=return_binary)
    
    # 保存结果
    cv2.imwrite(output_path, result)
    
    # 统计信息
    if return_binary:
        coverage = np.count_nonzero(result) / result.size * 100
        print(f"  📊 Edge coverage: {coverage:.1f}%")
    
    return True


def find_images(directory: str):
    """递归查找目录下的所有图片文件"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description='Laplacian边缘增强器')
    parser.add_argument('-r', '--round', help='轮次目录 (单文件模式)')
    parser.add_argument('-o', '--output-dir', help='output目录 (批量处理模式)')
    parser.add_argument('--gray', action='store_true', help='输出灰度图而非二值图')
    parser.add_argument('--suffix', default='_laplacian', help='输出文件后缀 (默认: _laplacian)')
    
    args = parser.parse_args()
    
    # 批量处理模式：处理 output 目录
    if args.output_dir:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            print(f"❌ Output directory not found: {output_dir}")
            return
        
        # 查找所有图片
        image_paths = find_images(output_dir)
        if not image_paths:
            print(f"❌ No images found in {output_dir}")
            return
        
        print(f"🔍 Found {len(image_paths)} images in {output_dir}")
        
        suffix = '_laplacian_gray.png' if args.gray else f'{args.suffix}.png'
        success_count = 0
        
        for image_path in image_paths:
            # 生成输出路径：在同一目录下，添加后缀
            path_obj = Path(image_path)
            output_path = path_obj.parent / f"{path_obj.stem}{suffix}"
            
            print(f"🎯 Processing: {image_path}")
            if process_single_image(image_path, str(output_path), return_binary=not args.gray):
                print(f"  💾 Saved: {output_path}")
                success_count += 1
            print()
        
        print(f"✅ Done! Processed {success_count}/{len(image_paths)} images")
        return
    
    # 单文件模式：处理指定轮次目录
    if args.round:
        round_dir = f'runs/{args.round}'
        
        # 查找输入图像
        input_path = None
        for ext in ['input.png', 'input.jpg', 'sample.png', 'sample.jpg']:
            path = os.path.join(round_dir, ext)
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            print(f"❌ No input image found in {round_dir}")
            return
        
        print(f"🎯 Processing: {input_path}")
        
        # 保存结果
        suffix = '_laplacian_gray.png' if args.gray else f'{args.suffix}.png'
        output_path = os.path.join(round_dir, f'lc_5{suffix}')
        
        if process_single_image(input_path, output_path, return_binary=not args.gray):
            print(f"💾 Saved: {output_path}")
        
        print("✅ Done!")
        return
    
    # 如果没有指定参数，显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()

