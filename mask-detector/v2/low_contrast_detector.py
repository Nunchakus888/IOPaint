"""
低对比度水印检测器 - 专门针对复杂背景下的半透明水印
==========================================================

核心问题：
- 水印颜色（灰白色）与背景（雪地）极其相近
- 水印透明度很低，对比度极低
- 常规边缘检测、阈值方法完全失效

核心理论：

1. 高通滤波增强法
   - 水印文字有高频边缘信息
   - 背景通常是平滑的低频信息
   - 高通滤波可以突出水印边缘
   
2. 多尺度高斯差分 (DoG)
   - 使用不同sigma的高斯模糊之差
   - 可以检测特定尺度的边缘
   - 文字笔画有特定的宽度范围

3. 局部自适应增强
   - 在极小的局部窗口内进行对比度增强
   - 即使全局对比度很低，局部可能仍有差异

4. Laplacian金字塔分解
   - 将图像分解为不同频率带
   - 水印可能在某个特定频率带更明显

5. 引导滤波分层
   - 将图像分为base层和detail层
   - 水印作为叠加层，可能在detail层更明显
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class LowContrastWatermarkDetector:
    """
    低对比度水印检测器
    
    专门针对：
    - 半透明水印
    - 与背景颜色相近的水印
    - 复杂纹理背景下的水印
    """
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        
    def detect(self, image: np.ndarray, output_dir: Optional[str] = None) -> np.ndarray:
        """
        主检测流程
        """
        h, w = image.shape[:2]
        
        print("=" * 60)
        print("低对比度水印检测器")
        print("=" * 60)
        
        # ===== 策略1: 高通滤波增强 =====
        hp_mask = self._highpass_enhancement(image)
        if self.debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'lc_1_highpass.png'), hp_mask)
        
        # ===== 策略2: 多尺度DoG =====
        dog_mask = self._multiscale_dog(image)
        if self.debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'lc_2_dog.png'), dog_mask)
        
        # ===== 策略3: 极端局部增强 =====
        local_mask = self._extreme_local_enhancement(image)
        if self.debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'lc_3_local.png'), local_mask)
        
        # ===== 策略4: 引导滤波分层 =====
        guided_mask = self._guided_filter_layer(image)
        if self.debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'lc_4_guided.png'), guided_mask)
        
        # ===== 策略5: Laplacian增强 =====
        lap_mask = self._laplacian_enhancement(image)
        if self.debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'lc_5_laplacian.png'), lap_mask)
        
        # ===== 融合策略 =====
        # 对于低对比度水印，放宽融合条件
        combined = (hp_mask.astype(np.float32) / 255 + 
                   dog_mask.astype(np.float32) / 255 + 
                   local_mask.astype(np.float32) / 255 +
                   guided_mask.astype(np.float32) / 255 +
                   lap_mask.astype(np.float32) / 255)
        
        # 至少1种方法检测到（放宽条件）
        final_mask = (combined >= 1.5).astype(np.uint8) * 255
        
        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 过滤大面积区域（可能是主体而不是水印）
        final_mask = self._filter_large_regions(final_mask, max_ratio=0.02)
        
        if self.debug and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'lc_6_combined.png'), final_mask)
            self._save_preview(image, final_mask, os.path.join(output_dir, 'detection_preview_lc.jpg'))
        
        coverage = np.count_nonzero(final_mask) / final_mask.size * 100
        print(f"💾 Final mask coverage: {coverage:.1f}%")
        
        return final_mask
    
    def _highpass_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        高通滤波增强
        
        原理：
        - 使用高斯模糊作为低通滤波器
        - 原图 - 低通 = 高通（高频细节）
        - 水印文字的边缘是高频信息
        
        为什么有效：
        - 平滑的背景在高通滤波后响应很弱
        - 文字边缘在高通滤波后响应很强
        """
        print("🔍 Strategy 1: High-pass filter enhancement...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 使用不同尺度的高通滤波
        masks = []
        
        for blur_size in [3, 7, 15]:
            # 低通：高斯模糊
            low_pass = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            
            # 高通：原图 - 低通
            high_pass = cv2.absdiff(gray, low_pass)
            
            # 归一化
            high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 自适应阈值
            thresh = cv2.adaptiveThreshold(
                high_pass, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, -2  # 负偏移量使阈值更敏感
            )
            masks.append(thresh)
        
        # 合并多尺度结果
        result = np.zeros_like(masks[0])
        for m in masks:
            result = cv2.bitwise_or(result, m)
        
        return result
    
    def _multiscale_dog(self, image: np.ndarray) -> np.ndarray:
        """
        多尺度高斯差分 (Difference of Gaussians)
        
        原理：
        - DoG ≈ Laplacian of Gaussian (LoG)
        - DoG(σ1, σ2) = G(σ1) - G(σ2)
        - 可以检测特定尺度的边缘和斑点
        
        为什么有效：
        - 文字笔画有特定的宽度（几个像素）
        - 选择合适的sigma可以精确匹配文字笔画宽度
        - 背景纹理通常没有这种特定尺度的结构
        """
        print("🔍 Strategy 2: Multi-scale Difference of Gaussians...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 多尺度DoG
        # sigma_pairs: (σ1, σ2)，检测大约 σ1~σ2 宽度的边缘
        sigma_pairs = [(1, 2), (1.5, 3), (2, 4), (3, 6)]
        
        masks = []
        for sigma1, sigma2 in sigma_pairs:
            # 计算高斯核大小（必须是奇数）
            k1 = int(6 * sigma1) | 1
            k2 = int(6 * sigma2) | 1
            
            g1 = cv2.GaussianBlur(gray, (k1, k1), sigma1)
            g2 = cv2.GaussianBlur(gray, (k2, k2), sigma2)
            
            # DoG
            dog = cv2.absdiff(g1, g2)
            
            # 归一化和阈值化
            dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, thresh = cv2.threshold(dog_norm, 15, 255, cv2.THRESH_BINARY)
            
            masks.append(thresh)
        
        # 合并结果
        result = np.zeros_like(masks[0])
        for m in masks:
            result = cv2.bitwise_or(result, m)
        
        return result
    
    def _extreme_local_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        极端局部增强
        
        原理：
        - 在非常小的局部窗口内进行对比度增强
        - 使用CLAHE（对比度受限自适应直方图均衡化）
        - clipLimit设置得很高以获得更强的增强效果
        
        为什么有效：
        - 即使全局对比度很低，在3x3或5x5的窗口内，
          水印与背景仍可能有微小的灰度差异
        - 极端的局部增强可以放大这些微小差异
        """
        print("🔍 Strategy 3: Extreme local enhancement...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 极端CLAHE参数
        # clipLimit高 → 对比度增强更强
        # tileGridSize小 → 局部化更强
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # 与原图做差分，突出被增强的区域
        diff = cv2.absdiff(enhanced, gray)
        
        # 归一化
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            diff_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 7, -1
        )
        
        # 另一种方法：使用极端的局部标准差
        local_std = self._compute_local_std(gray, kernel_size=3)
        std_norm = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # OTSU阈值
        _, std_thresh = cv2.threshold(std_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 合并两种方法
        result = cv2.bitwise_or(thresh, std_thresh)
        
        return result
    
    def _guided_filter_layer(self, image: np.ndarray) -> np.ndarray:
        """
        引导滤波分层
        
        原理：
        - 引导滤波可以将图像分解为base和detail两层
        - base层是平滑的大尺度结构（背景）
        - detail层是高频细节（水印、纹理）
        - 水印作为叠加层，在detail层可能更明显
        
        引导滤波公式：
        - q = a * I + b（局部线性模型）
        - 其中I是引导图像，这里使用图像本身
        
        为什么有效：
        - 引导滤波在边缘处理上优于高斯滤波
        - 可以更好地分离不同的图像层
        """
        print("🔍 Strategy 4: Guided filter layer separation...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # OpenCV的引导滤波
        # radius: 滤波半径
        # eps: 正则化参数，控制平滑程度
        
        # 获取base层（平滑层）
        radius = 8
        eps = 0.01 * 255 * 255  # 约650
        
        # 需要float32
        gray_float = gray.astype(np.float32) / 255.0
        
        # 引导滤波（使用图像本身作为引导）
        base = cv2.ximgproc.guidedFilter(
            guide=gray_float, 
            src=gray_float, 
            radius=radius, 
            eps=eps
        )
        
        # detail层 = 原图 - base层
        detail = gray_float - base
        
        # 增强detail层
        detail_enhanced = np.abs(detail) * 5  # 放大5倍
        detail_enhanced = np.clip(detail_enhanced, 0, 1)
        
        # 转换为uint8
        detail_uint8 = (detail_enhanced * 255).astype(np.uint8)
        
        # 阈值化
        _, thresh = cv2.threshold(detail_uint8, 20, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def _laplacian_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Laplacian增强检测
        
        原理：
        - Laplacian算子是二阶导数算子
        - 对边缘响应非常敏感
        - 可以检测快速变化的灰度区域（文字边缘）
        
        为什么有效：
        - 文字有锐利的边缘
        - 即使对比度很低，边缘的二阶导数仍不为零
        """
        print("🔍 Strategy 5: Laplacian enhancement...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 先轻微模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Laplacian算子
        # ksize=3: 3x3的Laplacian核
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        
        # 取绝对值
        laplacian_abs = np.abs(laplacian)
        
        # 归一化
        lap_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            lap_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, -2
        )
        
        return thresh
    
    def _compute_local_std(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """计算局部标准差"""
        image_float = image.astype(np.float64)
        kernel = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size ** 2)
        
        local_mean = cv2.filter2D(image_float, -1, kernel)
        local_sqr_mean = cv2.filter2D(image_float ** 2, -1, kernel)
        
        variance = local_sqr_mean - local_mean ** 2
        variance = np.maximum(variance, 0)
        local_std = np.sqrt(variance)
        
        return local_std.astype(np.float32)
    
    def _filter_large_regions(self, mask: np.ndarray, max_ratio: float = 0.02) -> np.ndarray:
        """
        过滤大面积区域
        
        原理：
        - 水印通常是小的文字区域
        - 大面积区域通常是主体或背景
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = mask.shape
        max_area = h * w * max_ratio
        
        filtered_mask = np.zeros_like(mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max_area and area > 10:  # 过滤过大和过小
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
        return filtered_mask
    
    def _save_preview(self, image: np.ndarray, mask: np.ndarray, output_path: str):
        """保存检测预览"""
        preview = image.copy()
        
        # 半透明红色覆盖
        overlay = preview.copy()
        overlay[mask > 127] = [0, 0, 255]
        preview = cv2.addWeighted(overlay, 0.4, preview, 0.6, 0)
        
        # 绿色轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 1)
        
        # 统计信息
        h, w = preview.shape[:2]
        coverage = np.count_nonzero(mask) / mask.size * 100
        
        cv2.rectangle(preview, (10, 10), (350, 80), (0, 0, 0), -1)
        cv2.putText(preview, f"Low Contrast Detector", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(preview, f"Regions: {len(contours)}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(preview, f"Coverage: {coverage:.1f}%", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, preview)
        print(f"📸 Preview saved: {output_path}")


def main():
    """测试低对比度检测器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='低对比度水印检测器')
    parser.add_argument('-r', '--round', required=True, help='测试轮次目录')
    parser.add_argument('--debug', action='store_true', help='保存中间调试结果')
    
    args = parser.parse_args()
    
    # 构建路径
    round_dir = f'runs/{args.round}'
    
    # 查找输入图像
    for ext in ['input.png', 'input.jpg', 'sample.png', 'sample.jpg']:
        input_path = os.path.join(round_dir, ext)
        if os.path.exists(input_path):
            break
    else:
        print(f"❌ No input image found in {round_dir}")
        return
    
    # 加载图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Failed to load image: {input_path}")
        return
    
    print(f"🎯 Processing: {input_path}")
    
    # 创建检测器
    detector = LowContrastWatermarkDetector(debug=args.debug)
    
    # 检测
    mask = detector.detect(image, output_dir=round_dir if args.debug else None)
    
    # 保存结果
    output_path = os.path.join(round_dir, 'mask_low_contrast.png')
    cv2.imwrite(output_path, mask)
    print(f"💾 Mask saved: {output_path}")
    
    print(f"✅ Detection completed!")


if __name__ == "__main__":
    main()
