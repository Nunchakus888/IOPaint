import cv2
import numpy as np
import os
from collections import defaultdict
import re

def scan_images_recursive(input_folder, pattern=r'.*x700\.(jpg|jpeg|png|bmp)$'):
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        dirs[:] = [d for d in dirs if d != 'post']
        for file in files:
            if re.match(pattern, file, re.IGNORECASE):
                image_paths.append(os.path.join(root, file))
    return image_paths

def filter_by_line_geometry(mask, angle_deg=25, line_width=25, min_density_ratio=0.2):
    """
    基于几何特征过滤噪音：水印是倾斜25度的平行多行文本
    行间的孤立像素必为噪音
    """
    h, w = mask.shape
    angle_rad = np.radians(angle_deg)
    sin_a, cos_a = np.sin(angle_rad), np.cos(angle_rad)
    
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    
    # 投影到垂直于行方向
    projections = (xs * sin_a + ys * cos_a).astype(int)
    proj_min, proj_max = projections.min(), projections.max()
    n = proj_max - proj_min + 1
    
    # 构建密度直方图
    hist = np.zeros(n, dtype=int)
    for p in projections:
        hist[p - proj_min] += 1
    
    # 滑动窗口统计密度
    kernel = np.ones(line_width)
    density = np.convolve(hist, kernel, mode='same')
    
    # 主阈值：用于中间区域
    threshold = density.max() * min_density_ratio
    
    # 标记有效区域
    valid_proj = density >= threshold
    
    # 边缘修复：首尾区域使用更宽松的阈值
    # 尾部范围要更大，因为右下角截断更严重
    head_range = line_width
    tail_range = line_width * 2  # 尾部用更大范围
    
    # 首部修复
    for i in range(min(head_range, n)):
        if density[i] >= threshold * 0.3:
            valid_proj[i] = True
    
    # 尾部修复（范围更大，阈值更低）
    for i in range(max(0, n - tail_range), n):
        if density[i] >= threshold * 0.1:  # 更宽松
            valid_proj[i] = True
    
    # 过滤
    clean_mask = np.zeros_like(mask)
    for x, y, p in zip(xs, ys, projections):
        if valid_proj[p - proj_min]:
            clean_mask[y, x] = 255
    
    return clean_mask

def extract_watermarks_by_group(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_images = scan_images_recursive(input_folder)
    groups = defaultdict(list)

    print(f"正在扫描... 共发现 {len(all_images)} 张符合条件的图片\n")
    
    for path in all_images:
        img = cv2.imread(path)
        if img is None: continue
        h, w = img.shape[:2]
        groups[(w, h)].append(path)

    # --- 核心处理循环 ---
    for (w, h), file_paths in groups.items():
        count = len(file_paths)
        group_name = f"{w}x{h}"
        
        print(f"--- 正在计算分组: {group_name} (样本数: {count}) ---")
        # print file_paths under group 
        # print(f"🔍 文件路径: {'\n'.join(file_paths)}")
        
        if count < 2:
            print(f"⚠️ 样本太少，跳过")
            continue

        # 初始化累加器 (使用 float32 记录连续能量信号)
        # 这里我们需要两个累加器，一个记录边缘，一个记录亮度突变
        accum_energy = np.zeros((h, w), dtype=np.float32)

        # CLAHE 用于在单图阶段增强微弱信号
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

        processed_count = 0
        for path in file_paths:
            img = cv2.imread(path)
            if img is None: continue
            
            # 1. 预处理：灰度 + CLAHE 强力拉伸对比度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)
            
            # --- 算法升级：双重特征提取 ---
            
            # 特征 A: 梯度 (Sobel) - 捕捉文字轮廓
            # 水印的边缘通常比背景（云、雾）更锐利
            gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            
            # 特征 B: 形态学 TopHat/BlackHat - 捕捉文字“实体”
            # TopHat 提取亮背景上的暗字，BlackHat 提取暗背景上的亮字
            # 水印通常比局部背景亮或暗，无论哪种，这两个运算都能提取出来
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)) # 文字笔画宽度大概的尺寸
            tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_morph)
            blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_morph)
            contrast_feat = cv2.add(tophat, blackhat) # 叠加亮字和暗字信号
            contrast_feat = contrast_feat.astype(np.float32)

            # 融合当前帧的能量 (梯度 + 对比度突变)
            # 这里的权重 0.5 可以调整，梯度负责边缘，contrast负责填满笔画内部
            current_energy = 0.4 * magnitude + 0.6 * contrast_feat
            
            # 累加到总图
            accum_energy += current_energy
            
            processed_count += 1
            # print(f"  > 已累加: {os.path.basename(path)}")

        # --- 后期合成 ---
        
        # 1. 归一化：将累加的巨大数值压缩回 0-255
        # 这一步非常神奇，因为背景是随机噪点，累加值低；文字是固定的，累加值极高
        result = cv2.normalize(accum_energy, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)

        # 2. 双阈值提取
        otsu_thresh, _ = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 常规阈值 → 斜向文字水印
        _, raw_mask = cv2.threshold(result, int(otsu_thresh), 255, cv2.THRESH_BINARY)
        text_mask = filter_by_line_geometry(raw_mask, angle_deg=25, line_width=25, min_density_ratio=0.2)
        
        # 高阈值 → 固定水印（每张图都有，累加信号极强）
        high_thresh = min(int(otsu_thresh * 1.8), 220)
        _, fixed_mask = cv2.threshold(result, high_thresh, 255, cv2.THRESH_BINARY)
        
        # 过滤异常大区域（固定水印通常面积适中）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fixed_mask, connectivity=8)
        max_area = w * h * 0.1  # 不超过图像面积的 10%
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                fixed_mask[labels == i] = 0
        
        # 3. 合并
        mask = cv2.bitwise_or(text_mask, fixed_mask)
        
        # 6. 闭运算 + 小区域过滤
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                clean_mask[labels == i] = 255
        mask = clean_mask

        # 保存结果
        output_filename = f"mask_{w}x{h}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, mask)
        
        print(f"✅ 完成！高质量Mask已保存至: {output_path}\n")

# --- 配置 ---
INPUT_DIR = 'enhance_analysis/images'
OUTPUT_DIR = 'enhance_analysis/masks'

extract_watermarks_by_group(INPUT_DIR, OUTPUT_DIR)