import cv2
import numpy as np
import os
from collections import defaultdict
import re

def scan_images_recursive(input_folder, pattern=r'.*x700\.(jpg|jpeg|png|bmp)$'):
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        dirs[:] = [d for d in dirs if d != 'post'] # 跳过特定目录
        for file in files:
            if re.match(pattern, file, re.IGNORECASE):
                image_paths.append(os.path.join(root, file))
    return image_paths

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

        # 2. 阈值提取
        # 因为我们累加了连续信号，这里的信噪比非常高
        # 使用 OTSU 自动阈值通常效果最好，如果觉得噪点多，可以换成 cv2.threshold(result, 60, 255, ...)
        _, mask = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 3. 关键修复：让文字更饱满，不缺失
        # 之前的代码可能腐蚀过度，这里我们做一点点“膨胀”来连接断点
        
        # 步骤 3.1: 移除微小噪点 (背景里的星星点点)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        # 步骤 3.2: 膨胀 (Dilation) - 让细笔画变粗，连接断裂处
        # 3x3 的核比较温和
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        
        # 步骤 3.3: 闭运算 (Closing) - 填补文字内部的空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        # 保存结果
        output_filename = f"mask_{w}x{h}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, mask)
        
        print(f"✅ 完成！高质量Mask已保存至: {output_path}\n")

# --- 配置 ---
INPUT_DIR = 'enhance_analysis/images'
OUTPUT_DIR = 'enhance_analysis/runs/extract_watermark'

extract_watermarks_by_group(INPUT_DIR, OUTPUT_DIR)