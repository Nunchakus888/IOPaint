"""
周期性水印检测器 - 优化版
==========================================================

核心算法：
1. 高通滤波增强水印（已验证有效）
2. 霍夫变换检测主方向（排除水平/垂直干扰）
3. 直接使用增强图像生成mask
4. 基于方向和周期的外推填充（提高召回率）
5. 主体保护（避免误检）

关键改进：
- 增强图像中水印已经清晰可见
- 利用周期性特征沿方向外推，大幅提高召回率
- 支持预览模式和自动水印去除
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os
import subprocess
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class PeriodicWatermarkDetector:
    """
    周期性水印检测器
    
    利用水印的周期性重复特征：
    - 水印沿特定方向（通常-30°~-45°）周期排列
    - 找到种子区域后，沿方向和周期外推填充
    """
    
    def __init__(self, enable_preview: bool = True):
        self.enable_preview = enable_preview
        self.detected_angle = 0.0
        self.detected_period = 80.0
        
    def detect(self, image: np.ndarray, output_dir: Optional[str] = None) -> np.ndarray:
        """
        主检测流程
        
        Args:
            image: BGR格式输入图像
            output_dir: 输出目录（用于保存中间结果和预览）
            
        Returns:
            二值mask (255=水印, 0=背景)
        """
        print("=" * 60)
        print("周期性水印检测器")
        print("=" * 60)
        
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ===== Step 1: 高通滤波增强水印 =====
        enhanced = self._enhance_watermark(gray)
        if self.enable_preview and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'step1_enhanced.png'), enhanced)
        
        # ===== Step 2: 检测水印方向 =====
        self.detected_angle = self._detect_direction(enhanced)
        print(f"📐 Detected angle: {self.detected_angle:.1f}°")
        
        # ===== Step 3: 分析周期 =====
        self.detected_period = self._analyze_period(enhanced, self.detected_angle)
        print(f"📏 Detected period: {self.detected_period:.1f} pixels")
        
        # ===== Step 4: 从增强图像生成种子mask =====
        seed_mask = self._enhanced_to_mask(enhanced)
        if self.enable_preview and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'step4_seed_mask.png'), seed_mask)
        
        # ===== Step 5: 基于方向和周期外推填充 =====
        extrapolated_mask = self._extrapolate_by_direction(
            enhanced, seed_mask, self.detected_angle, self.detected_period
        )
        if self.enable_preview and output_dir:
            cv2.imwrite(os.path.join(output_dir, 'step5_extrapolated.png'), extrapolated_mask)
        
        # ===== Step 6: 主体保护 =====
        final_mask = self._protect_subject(image, extrapolated_mask)
        
        # ===== Step 7: 最终形态学优化 =====
        final_mask = self._final_morphology(final_mask)
        
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, 'mask.png'), final_mask)
            if self.enable_preview:
                self._save_preview(image, final_mask, self.detected_angle, 
                                 os.path.join(output_dir, 'detection_preview.jpg'))
        
        coverage = np.count_nonzero(final_mask) / final_mask.size * 100
        print(f"💾 Final coverage: {coverage:.1f}%")
        
        return final_mask
    
    def _enhance_watermark(self, gray: np.ndarray) -> np.ndarray:
        """高通滤波增强水印"""
        print("🔍 Step 1: Enhancing watermark...")
        
        enhanced = np.zeros_like(gray, dtype=np.float32)
        for blur_size in [7, 15, 25]:
            blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            highpass = cv2.absdiff(gray, blur)
            enhanced += highpass.astype(np.float32)
        
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        
        return enhanced
    
    def _detect_direction(self, enhanced: np.ndarray) -> float:
        """使用霍夫变换检测水印方向"""
        print("🔍 Step 2: Detecting direction...")
        
        edges = cv2.Canny(enhanced, 30, 80)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)
        
        if lines is None:
            return -30.0
        
        angle_counts = np.zeros(180)
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi - 90
            angle_idx = int(angle + 90) % 180
            angle_counts[angle_idx] += 1
        
        # 排除水平和垂直
        exclude = 15
        angle_counts[90-exclude:90+exclude] = 0
        angle_counts[:exclude] = 0
        angle_counts[180-exclude:] = 0
        
        angle_counts = gaussian_filter1d(angle_counts, sigma=3)
        peaks, _ = find_peaks(angle_counts, height=np.max(angle_counts) * 0.3)
        
        if len(peaks) > 0:
            best_peak = peaks[np.argmax(angle_counts[peaks])]
            return best_peak - 90
        
        return -30.0
    
    def _analyze_period(self, enhanced: np.ndarray, angle: float) -> float:
        """分析水印的重复周期"""
        print("🔍 Step 3: Analyzing period...")
        
        h, w = enhanced.shape
        angle_rad = angle * np.pi / 180
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        perp_dx, perp_dy = -dy, dx
        
        max_shift = 200
        accumulated_autocorr = np.zeros(max_shift)
        count = 0
        center_y, center_x = h // 2, w // 2
        
        for offset in range(-150, 151, 10):
            start_x = center_x + offset * perp_dx
            start_y = center_y + offset * perp_dy
            
            profile = []
            for t in range(-max_shift, max_shift):
                x, y = int(start_x + t * dx), int(start_y + t * dy)
                if 0 <= x < w and 0 <= y < h:
                    profile.append(float(enhanced[y, x]))
            
            if len(profile) > max_shift:
                profile = np.array(profile) - np.mean(profile)
                if np.std(profile) > 1:
                    autocorr = np.correlate(profile, profile, mode='full')
                    autocorr = autocorr[len(autocorr)//2:][:max_shift]
                    if autocorr[0] > 0:
                        autocorr = autocorr / autocorr[0]
                        accumulated_autocorr += autocorr
                        count += 1
        
        if count > 0:
            accumulated_autocorr /= count
        
        min_period = 40
        peaks, _ = find_peaks(accumulated_autocorr[min_period:], distance=20, height=0.1)
        
        return float(peaks[0] + min_period) if len(peaks) > 0 else 80.0
    
    def _enhanced_to_mask(self, enhanced: np.ndarray) -> np.ndarray:
        """从增强图像生成种子mask"""
        print("🔍 Step 4: Generating seed mask...")
        
        mean_val = np.mean(enhanced)
        std_val = np.std(enhanced)
        
        high_thresh = int(mean_val + 1.0 * std_val)
        mid_thresh = int(mean_val + 0.5 * std_val)
        low_thresh = int(mean_val + 0.2 * std_val)
        
        print(f"   Thresholds: low={low_thresh}, mid={mid_thresh}, high={high_thresh}")
        
        _, thresh_high = cv2.threshold(enhanced, high_thresh, 255, cv2.THRESH_BINARY)
        _, thresh_mid = cv2.threshold(enhanced, mid_thresh, 255, cv2.THRESH_BINARY)
        _, thresh_low = cv2.threshold(enhanced, low_thresh, 255, cv2.THRESH_BINARY)
        
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, -5
        )
        edges = cv2.Canny(enhanced, 30, 80)
        
        combined = thresh_high.copy()
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(thresh_mid, edges))
        combined = cv2.bitwise_or(combined, cv2.bitwise_and(thresh_low, adaptive))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return combined
    
    def _extrapolate_by_direction(self, enhanced: np.ndarray, seed_mask: np.ndarray,
                                    angle: float, period: float) -> np.ndarray:
        """基于方向和周期外推填充水印区域"""
        print("🔍 Step 5: Extrapolating by direction and period...")
        
        h, w = enhanced.shape
        angle_rad = angle * np.pi / 180
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        perp_dx, perp_dy = -dy, dx
        
        effective_period = max(period, 60.0)
        row_spacing = effective_period * 0.6
        
        contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        seed_points = []
        seed_energies = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                x1, y1 = max(0, cx-20), max(0, cy-20)
                x2, y2 = min(w, cx+20), min(h, cy+20)
                energy = np.sum(enhanced[y1:y2, x1:x2] > 15)
                if energy > 20:
                    seed_points.append((cx, cy))
                    seed_energies.append(energy)
        
        print(f"   Found {len(seed_points)} seed points")
        
        if len(seed_points) == 0:
            return seed_mask
        
        avg_energy = np.mean(seed_energies) if seed_energies else 100
        energy_threshold = avg_energy * 0.1
        
        extrapolated = seed_mask.copy()
        max_steps_main = 40
        max_steps_perp = 20
        search_radius = 25
        
        for cx, cy in seed_points:
            for direction in [1, -1]:
                for step in range(1, max_steps_main):
                    new_x = int(cx + direction * step * effective_period * dx)
                    new_y = int(cy + direction * step * effective_period * dy)
                    
                    if not (10 <= new_x < w-10 and 10 <= new_y < h-10):
                        break
                    
                    x1, y1 = max(0, new_x-search_radius), max(0, new_y-search_radius)
                    x2, y2 = min(w, new_x+search_radius), min(h, new_y+search_radius)
                    region_energy = np.sum(enhanced[y1:y2, x1:x2] > 12)
                    
                    if region_energy > energy_threshold:
                        local_region = enhanced[y1:y2, x1:x2]
                        _, local_mask = cv2.threshold(local_region, 15, 255, cv2.THRESH_BINARY)
                        extrapolated[y1:y2, x1:x2] = cv2.bitwise_or(
                            extrapolated[y1:y2, x1:x2], local_mask
                        )
            
            for direction in [1, -1]:
                for step in range(1, max_steps_perp):
                    new_x = int(cx + direction * step * row_spacing * perp_dx)
                    new_y = int(cy + direction * step * row_spacing * perp_dy)
                    
                    if not (10 <= new_x < w-10 and 10 <= new_y < h-10):
                        break
                    
                    x1, y1 = max(0, new_x-search_radius), max(0, new_y-search_radius)
                    x2, y2 = min(w, new_x+search_radius), min(h, new_y+search_radius)
                    region_energy = np.sum(enhanced[y1:y2, x1:x2] > 12)
                    
                    if region_energy > energy_threshold:
                        local_region = enhanced[y1:y2, x1:x2]
                        _, local_mask = cv2.threshold(local_region, 15, 255, cv2.THRESH_BINARY)
                        extrapolated[y1:y2, x1:x2] = cv2.bitwise_or(
                            extrapolated[y1:y2, x1:x2], local_mask
                        )
        
        return extrapolated
    
    def _protect_subject(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """主体保护"""
        print("🔍 Step 6: Protecting subject...")
        
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        protected_mask = np.zeros_like(mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / (h * w)
            
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            dist_ratio = np.sqrt((cx - center_x)**2 + (cy - center_y)**2) / max_dist
            
            is_large_center = (area_ratio > 0.05 and dist_ratio < 0.2)
            
            if not is_large_center:
                cv2.drawContours(protected_mask, [cnt], -1, 255, -1)
        
        return protected_mask
    
    def _final_morphology(self, mask: np.ndarray) -> np.ndarray:
        """最终形态学优化"""
        print("🔍 Step 7: Final morphology optimization...")
        
        h, w = mask.shape
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        result = cv2.dilate(result, kernel_dilate, iterations=1)
        
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(result)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / (h * w)
            if area >= 15 and area_ratio < 0.05:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        
        return filtered
    
    def _save_preview(self, image: np.ndarray, mask: np.ndarray,
                     angle: float, output_path: str):
        """保存检测预览图"""
        preview = image.copy()
        h, w = preview.shape[:2]
        
        overlay = preview.copy()
        overlay[mask > 127] = [0, 0, 255]
        preview = cv2.addWeighted(overlay, 0.5, preview, 0.5, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 1)
        
        cx, cy = w // 2, h // 2
        angle_rad = angle * np.pi / 180
        line_len = min(h, w) // 3
        x1, y1 = int(cx - line_len * np.cos(angle_rad)), int(cy - line_len * np.sin(angle_rad))
        x2, y2 = int(cx + line_len * np.cos(angle_rad)), int(cy + line_len * np.sin(angle_rad))
        cv2.line(preview, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        coverage = np.count_nonzero(mask) / mask.size * 100
        
        cv2.rectangle(preview, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(preview, "Periodic Watermark Detector", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(preview, f"Angle: {angle:.1f} deg | Period: {self.detected_period:.0f}px", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(preview, f"Regions: {len(contours)}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(preview, f"Coverage: {coverage:.1f}%", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, preview)
        print(f"📸 Preview saved: {output_path}")
    
    def generate_mask(self, image: np.ndarray, output_dir: Optional[str] = None) -> np.ndarray:
        """生成水印mask的简化接口"""
        return self.detect(image, output_dir)


def run_watermark_removal(round_dir: str, input_image: str, mask_file: str):
    """运行水印去除命令"""
    import shutil
    import tempfile
    
    try:
        result = subprocess.run([
            "conda", "run", "-n", "py312aiwatermark",
            "iopaint", "--help"
        ], capture_output=True, text=True, timeout=10)
        iopaint_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        iopaint_available = False

    if not iopaint_available:
        print("⚠️ iopaint not available. Skipping automatic removal.")
        print(f"💡 Manual: iopaint run --model=lama --device=cpu --image={input_image} --mask={mask_file} --output={round_dir}")
        return

    input_image_path = os.path.join(round_dir, 'input.jpg')
    if not os.path.exists(input_image_path):
        input_image_path = os.path.join(round_dir, 'input.png')
    if not os.path.exists(input_image_path):
        input_image_path = input_image
    
    input_image_path = os.path.abspath(input_image_path)
    mask_file = os.path.abspath(mask_file)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "conda", "run", "-n", "py312aiwatermark",
            "env", "KMP_DUPLICATE_LIB_OK=TRUE",
            "iopaint", "run",
            "--model=lama", "--device=cpu",
            f"--image={input_image_path}",
            f"--mask={mask_file}",
            f"--output={temp_dir}"
        ]

        print(f"🔧 Running iopaint...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                output_files = [f for f in os.listdir(temp_dir) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                if output_files:
                    temp_output = os.path.join(temp_dir, output_files[0])
                    ext = os.path.splitext(output_files[0])[1]
                    final_output = os.path.join(round_dir, f'output{ext}')
                    shutil.copy2(temp_output, final_output)
                    print(f"✨ Watermark removal completed!")
                    print(f"   Output: {os.path.basename(final_output)}")
                else:
                    print("⚠️ No output files generated")
            else:
                print(f"❌ Removal failed (exit code: {result.returncode})")
        except subprocess.TimeoutExpired:
            print("⏰ Removal timed out (5 minutes)")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='周期性水印检测器')
    parser.add_argument('-r', '--round', required=True, help='轮次目录 (如: 15, test1)')
    parser.add_argument('--preview', action='store_true', default=True, help='生成检测预览图 (默认开启)')
    parser.add_argument('--no-preview', action='store_true', help='禁用预览')
    parser.add_argument('--remove', action='store_true', help='自动运行水印去除')
    parser.add_argument('--debug', action='store_true', help='保存所有中间步骤结果')
    
    args = parser.parse_args()
    
    round_dir = f'runs/{args.round}'
    os.makedirs(round_dir, exist_ok=True)
    
    # 查找输入图片：先查找round_dir，再查找当前目录
    input_path = None
    
    # 1. 先在round_dir中查找
    for ext in ['input.png', 'input.jpg', 'sample.png', 'sample.jpg']:
        path = os.path.join(round_dir, ext)
        if os.path.exists(path):
            input_path = path
            break
    
    # 2. 如果没找到，在当前目录（mask-detector/v2/）查找
    if input_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for ext in ['input.png', 'input.jpg', 'sample.png', 'sample.jpg']:
            path = os.path.join(script_dir, ext)
            if os.path.exists(path):
                input_path = path
                break
    
    if input_path is None:
        print(f"❌ No input image found in {round_dir} or current directory")
        print(f"   Please place input.png/input.jpg in {round_dir}/ or current directory")
        return
    
    # 如果从当前目录读取，复制到round_dir以便后续使用
    if not input_path.startswith(round_dir):
        import shutil
        input_ext = os.path.splitext(input_path)[1]
        round_input_path = os.path.join(round_dir, f'input{input_ext}')
        shutil.copy2(input_path, round_input_path)
        print(f"📋 Copied input image to {round_input_path}")
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Failed to load: {input_path}")
        return
    
    print(f"🎯 Processing: {input_path}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    enable_preview = not args.no_preview
    detector = PeriodicWatermarkDetector(enable_preview=enable_preview)
    
    output_dir = round_dir if (args.debug or enable_preview) else None
    mask = detector.detect(image, output_dir=output_dir)
    
    mask_path = os.path.join(round_dir, 'mask.png')
    cv2.imwrite(mask_path, mask)
    print(f"💾 Mask saved: {mask_path}")
    
    if args.remove:
        print("\n🧹 Starting automatic watermark removal...")
        run_watermark_removal(round_dir, input_path, mask_path)
    
    print(f"\n✅ Round {args.round} completed!")


if __name__ == "__main__":
    main()
