"""
水印增强分析工具 - 通过多种增强方法分析水印特征
==============================================

输出多种增强效果，分析哪种能让规律重复的水印特征更明显
"""

import cv2
import numpy as np
import os


def enhance_watermark(image_path: str, output_dir: str):
    """多种增强方法分析水印特征"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"📐 Image size: {w}x{h}")
    
    # ===== 1. 高通滤波 (去除低频背景，保留水印边缘) =====
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    highpass = cv2.subtract(gray, blur)
    highpass_norm = cv2.normalize(highpass, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, '01_highpass.png'), highpass_norm)
    print("✅ 1. 高通滤波 - 去除低频背景")
    
    # ===== 2. 拉普拉斯算子 (边缘增强) =====
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian)
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '02_laplacian.png'), laplacian_norm)
    print("✅ 2. 拉普拉斯算子 - 边缘增强")
    
    # ===== 3. CLAHE 局部对比度增强 =====
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, '03_clahe.png'), clahe_result)
    print("✅ 3. CLAHE - 局部对比度增强")
    
    # ===== 4. Sobel 梯度幅值 =====
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '04_sobel.png'), sobel_norm)
    print("✅ 4. Sobel 梯度幅值")
    
    # ===== 5. 局部标准差 (纹理变化图) =====
    kernel_size = 5
    mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray.astype(np.float32)**2, (kernel_size, kernel_size))
    std_dev = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    std_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '05_local_std.png'), std_norm)
    print("✅ 5. 局部标准差 - 纹理变化图")
    
    # ===== 6. 傅里叶频谱 (检测周期性模式) =====
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '06_fft_spectrum.png'), magnitude_norm)
    print("✅ 6. 傅里叶频谱 - 周期性模式")
    
    # ===== 7. 形态学梯度 =====
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(os.path.join(output_dir, '07_morph_gradient.png'), morph_grad)
    print("✅ 7. 形态学梯度")
    
    # ===== 8. 高通 + CLAHE 组合 =====
    highpass_clahe = clahe.apply(highpass_norm)
    cv2.imwrite(os.path.join(output_dir, '08_highpass_clahe.png'), highpass_clahe)
    print("✅ 8. 高通 + CLAHE 组合")
    
    # ===== 9. 自适应阈值 =====
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_dir, '09_adaptive_thresh.png'), adaptive)
    print("✅ 9. 自适应阈值")
    
    # ===== 10. 高通滤波后自适应阈值 =====
    highpass_adaptive = cv2.adaptiveThreshold(highpass_norm, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_dir, '10_highpass_adaptive.png'), highpass_adaptive)
    print("✅ 10. 高通 + 自适应阈值")
    
    # ===== 11. 不同尺度高通滤波 =====
    for i, ksize in enumerate([7, 15, 31, 51], start=1):
        blur_k = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        hp = cv2.subtract(gray, blur_k)
        hp_norm = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(output_dir, f'11_highpass_k{ksize}.png'), hp_norm)
    print("✅ 11. 多尺度高通滤波 (k=7,15,31,51)")
    
    # ===== 12. 频域高通滤波 =====
    # 创建高通滤波器 (中心为0，边缘为1)
    crow, ccol = h // 2, w // 2
    mask_fft = np.ones((h, w), np.float32)
    r = 30  # 截止半径
    cv2.circle(mask_fft, (ccol, crow), r, 0, -1)
    
    fshift_filtered = fshift * mask_fft
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    fft_highpass = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '12_fft_highpass.png'), fft_highpass)
    print("✅ 12. 频域高通滤波")
    
    # ===== 13. 颜色通道差异 (检测颜色异常) =====
    b, g, r = cv2.split(image)
    # 计算各通道与灰度的差异
    color_diff = cv2.absdiff(r, g) + cv2.absdiff(g, b) + cv2.absdiff(r, b)
    color_diff_norm = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, '13_color_diff.png'), color_diff_norm)
    print("✅ 13. 颜色通道差异")
    
    print(f"\n📁 所有增强结果已保存到: {output_dir}")
    print("\n🔍 分析建议:")
    print("   - 查看哪种增强方法让水印文字最清晰")
    print("   - 观察水印的规律重复模式")
    print("   - 频谱图中的规律亮点表示周期性")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='水印增强分析')
    parser.add_argument('-i', '--input', default='14_x700.JPG', help='输入图像')
    parser.add_argument('-o', '--output', default='enhance_analysis', help='输出目录')
    
    args = parser.parse_args()
    
    enhance_watermark(args.input, args.output)

