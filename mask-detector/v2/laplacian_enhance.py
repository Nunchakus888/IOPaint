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


def main():
    parser = argparse.ArgumentParser(description='Laplacian边缘增强器')
    parser.add_argument('-r', '--round', required=True, help='轮次目录')
    parser.add_argument('--gray', action='store_true', help='输出灰度图而非二值图')
    
    args = parser.parse_args()
    
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
    
    # 加载图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Failed to load: {input_path}")
        return
    
    print(f"🎯 Processing: {input_path}")
    
    # 生成Laplacian增强图
    result = laplacian_enhance(image, return_binary=not args.gray)
    
    # 保存结果
    suffix = '_laplacian_gray.png' if args.gray else '_laplacian.png'
    output_path = os.path.join(round_dir, f'lc_5{suffix}')
    cv2.imwrite(output_path, result)
    print(f"💾 Saved: {output_path}")
    
    # 统计信息
    if not args.gray:
        coverage = np.count_nonzero(result) / result.size * 100
        print(f"📊 Edge coverage: {coverage:.1f}%")
    
    print("✅ Done!")


if __name__ == "__main__":
    main()

