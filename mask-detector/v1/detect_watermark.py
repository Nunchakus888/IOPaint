#!/usr/bin/env python3
"""
水印自动检测和提取工具
从样本图片中检测并提取固定位置的水印
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class WatermarkDetector:
    """水印检测器"""
    
    def __init__(self, image_path: Path):
        """
        初始化水印检测器
        
        Args:
            image_path: 样本图片路径
        """
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape[:2]
        
        # 生成增强版灰度图，用于检测半透明水印
        self.gray_enhanced = self._enhance_for_watermark()
    
    def _enhance_for_watermark(self) -> np.ndarray:
        """
        增强图像以提高半透明水印的可见性
        
        Returns:
            增强后的灰度图
        """
        # 1. CLAHE (对比度限制自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(self.gray)
        
        # 2. 锐化以增强边缘
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 3. 伽马校正（提亮暗部）
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        gamma_corrected = cv2.LUT(sharpened, table)
        
        return gamma_corrected
    
    def detect_corner_watermarks(
        self,
        corner_size_ratio: float = 0.25,
        threshold: int = 127
    ) -> List[Dict]:
        """
        检测四个角落的水印
        
        Args:
            corner_size_ratio: 角落区域大小（占图片宽高的比例）
            threshold: 二值化阈值
        
        Returns:
            检测到的水印区域列表
        """
        watermarks = []
        
        # 计算角落区域大小
        corner_w = int(self.width * corner_size_ratio)
        corner_h = int(self.height * corner_size_ratio)
        
        # 定义四个角落区域
        corners = {
            'top_left': (0, 0, corner_w, corner_h),
            'top_right': (self.width - corner_w, 0, self.width, corner_h),
            'bottom_left': (0, self.height - corner_h, corner_w, self.height),
            'bottom_right': (self.width - corner_w, self.height - corner_h, self.width, self.height)
        }
        
        for corner_name, (x1, y1, x2, y2) in corners.items():
            # 提取角落区域
            corner_region = self.gray[y1:y2, x1:x2]
            
            # 检测是否有内容（通过边缘检测）
            edges = cv2.Canny(corner_region, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            if edge_density > 0.01:  # 如果边缘密度超过阈值，认为有水印
                # 找到实际内容的边界框
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 合并所有轮廓的边界框
                    all_points = np.vstack([cv2.boundingRect(c) for c in contours])
                    min_x = all_points[:, 0].min()
                    min_y = all_points[:, 1].min()
                    max_x = (all_points[:, 0] + all_points[:, 2]).max()
                    max_y = (all_points[:, 1] + all_points[:, 3]).max()
                    
                    # 添加边距
                    margin = 10
                    actual_x1 = max(0, x1 + min_x - margin)
                    actual_y1 = max(0, y1 + min_y - margin)
                    actual_x2 = min(self.width, x1 + max_x + margin)
                    actual_y2 = min(self.height, y1 + max_y + margin)
                    
                    watermarks.append({
                        'position': corner_name,
                        'bbox': (actual_x1, actual_y1, actual_x2, actual_y2),
                        'relative_bbox': (
                            actual_x1 / self.width,
                            actual_y1 / self.height,
                            actual_x2 / self.width,
                            actual_y2 / self.height
                        ),
                        'edge_density': edge_density
                    })
        
        return watermarks
    
    def detect_text_regions(self) -> List[Dict]:
        """
        使用形态学操作检测文字区域（水印通常是文字）
        支持多方向检测，包括斜向水印，增强对半透明水印的检测
        
        Returns:
            检测到的文字区域列表
        """
        text_regions = []
        
        # 多阈值二值化策略 - 同时使用原图和增强图
        binary_images = []
        
        # 使用增强图像进行检测（对半透明水印更敏感）
        # 1. Otsu自动阈值 (增强图)
        _, binary_otsu_enh = cv2.threshold(self.gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_images.append(binary_otsu_enh)
        
        # 2. Otsu反向 (增强图)
        _, binary_otsu_inv_enh = cv2.threshold(self.gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_images.append(binary_otsu_inv_enh)
        
        # 3. 自适应阈值（增强图，更适合光照不均匀的情况）
        binary_adaptive_enh = cv2.adaptiveThreshold(
            self.gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 5
        )
        binary_images.append(binary_adaptive_enh)
        
        # 4. 自适应阈值反向（增强图）
        binary_adaptive_inv_enh = cv2.adaptiveThreshold(
            self.gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        binary_images.append(binary_adaptive_inv_enh)
        
        # 5. 多级阈值（检测不同透明度的水印）
        for threshold in [140, 160, 180, 200]:
            _, binary_thresh = cv2.threshold(self.gray_enhanced, threshold, 255, cv2.THRESH_BINARY)
            binary_images.append(binary_thresh)
            _, binary_thresh_inv = cv2.threshold(self.gray_enhanced, threshold, 255, cv2.THRESH_BINARY_INV)
            binary_images.append(binary_thresh_inv)
        
        # 使用原图进行补充检测
        _, binary_otsu = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_images.append(binary_otsu)
        
        _, binary_low = cv2.threshold(self.gray, 80, 255, cv2.THRESH_BINARY_INV)
        binary_images.append(binary_low)
        
        # 定义多个方向的形态学核（适应不同角度的水印）
        kernels = [
            # 横向
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5)),
            # 纵向
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20)),
            # 斜向 - 通过旋转矩形来近似
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            # 十字形（适合各种方向）
            cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11)),
        ]
        
        detected_regions = []
        
        # 对每种二值化结果和每种kernel进行检测
        for binary in binary_images:
            for kernel in kernels:
                # 形态学闭运算，连接文字
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                dilated = cv2.dilate(closed, kernel, iterations=1)
                
                # 查找轮廓
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 过滤太小或太大的区域
                    area = w * h
                    if area < 300 or area > self.width * self.height * 0.6:  # 降低最小面积
                        continue
                    
                    # 放宽长宽比限制（支持斜向和纵向）
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if aspect_ratio < 1.2:  # 太方正，可能不是文字
                        continue
                    
                    # 检查是否在边缘区域或中心重复（水印位置）
                    margin = 80  # 增加边缘容忍度
                    center_x, center_y = x + w // 2, y + h // 2
                    
                    is_near_edge = (
                        x < margin or y < margin or
                        x + w > self.width - margin or
                        y + h > self.height - margin
                    )
                    
                    # 检查是否在中心区域（对角线水印）
                    is_in_center_zone = (
                        self.width * 0.2 < center_x < self.width * 0.8 and
                        self.height * 0.2 < center_y < self.height * 0.8
                    )
                    
                    if is_near_edge or is_in_center_zone:
                        detected_regions.append({
                            'bbox': (x, y, x + w, y + h),
                            'relative_bbox': (
                                x / self.width,
                                y / self.height,
                                (x + w) / self.width,
                                (y + h) / self.height
                            ),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        # 去重（同一区域可能被多次检测）
        if detected_regions:
            text_regions = self._merge_overlapping_regions(detected_regions, iou_threshold=0.3)
        
        return text_regions
    
    def detect_rotated_text(self, angles: List[int] = None) -> List[Dict]:
        """
        通过旋转图像检测斜向文字水印（优化版）
        
        Args:
            angles: 要检测的角度列表（度）
        
        Returns:
            检测到的斜向文字区域
        """
        if angles is None:
            # 扩展角度范围，更细粒度
            angles = [-45, -40, -35, -30, -25, -20, -15, 15, 20, 25, 30, 35, 40, 45]
        
        rotated_regions = []
        h, w = self.gray_enhanced.shape
        
        print(f"    Detecting angles: {angles}")
        
        for angle in angles:
            # 旋转增强图像（对半透明水印更敏感）
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 计算新的边界大小
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # 调整旋转矩阵
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # 执行旋转 - 使用增强图像
            rotated = cv2.warpAffine(self.gray_enhanced, rotation_matrix, (new_w, new_h), 
                                     borderValue=255)
            
            # 多种二值化方法
            binaries = []
            
            # 1. Otsu阈值
            _, binary1 = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binaries.append(binary1)
            
            # 2. 反向Otsu（检测深色水印）
            _, binary2 = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binaries.append(binary2)
            
            # 3. 固定阈值（检测半透明水印）
            _, binary3 = cv2.threshold(rotated, 160, 255, cv2.THRESH_BINARY)
            binaries.append(binary3)
            
            for binary in binaries:
                # 使用横向kernel连接字符
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
                dilated = cv2.dilate(binary, kernel, iterations=1)
                
                # 再次膨胀以连接相邻的文字
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
                dilated = cv2.dilate(dilated, kernel2, iterations=1)
                
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 反向旋转矩阵（用于坐标转换）
                inv_rotation_matrix = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), -angle, 1.0)
                
                for contour in contours:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    area = w_box * h_box
                    
                    # 放宽过滤条件，检测更小的水印
                    if area < 200 or area > new_w * new_h * 0.5:
                        continue
                    
                    # 降低长宽比要求，适应更多文字排列
                    aspect_ratio = w_box / h_box if h_box > 0 else 0
                    if aspect_ratio < 1.5:  # 降低要求
                        continue
                    
                    # 计算旋转后的包围框并转换回原坐标系
                    box_points = np.array([
                        [x, y],
                        [x + w_box, y],
                        [x + w_box, y + h_box],
                        [x, y + h_box]
                    ], dtype=np.float32)
                    
                    ones = np.ones((4, 1))
                    box_points_homogeneous = np.hstack([box_points, ones])
                    original_points = inv_rotation_matrix.dot(box_points_homogeneous.T).T
                    
                    orig_x = max(0, int(original_points[:, 0].min()))
                    orig_y = max(0, int(original_points[:, 1].min()))
                    orig_x2 = min(self.width, int(original_points[:, 0].max()))
                    orig_y2 = min(self.height, int(original_points[:, 1].max()))
                    
                    if orig_x2 > orig_x and orig_y2 > orig_y:
                        rotated_regions.append({
                            'bbox': (orig_x, orig_y, orig_x2, orig_y2),
                            'relative_bbox': (
                                orig_x / self.width,
                                orig_y / self.height,
                                orig_x2 / self.width,
                                orig_y2 / self.height
                            ),
                            'area': (orig_x2 - orig_x) * (orig_y2 - orig_y),
                            'angle': angle,
                            'confidence': aspect_ratio  # 用长宽比作为置信度
                        })
        
        return rotated_regions
    
    def detect_mser_text(self) -> List[Dict]:
        """
        使用MSER (Maximally Stable Extremal Regions) 检测文字
        MSER对不同字体、大小、方向的文字都很鲁棒，特别适合半透明水印
        
        Returns:
            检测到的文字区域
        """
        mser_regions = []
        
        try:
            # 创建MSER检测器（参数调整为更敏感，适应半透明水印）
            mser = cv2.MSER_create(
                _delta=2,           # 降低delta以检测更细微的变化
                _min_area=15,       # 降低最小面积
                _max_area=int(self.width * self.height * 0.4),
                _max_variation=0.7, # 提高variation容忍度
                _min_diversity=0.05,# 降低diversity要求
                _max_evolution=250,
                _area_threshold=1.005,
                _min_margin=0.0005,
                _edge_blur_size=3
            )
            
            # 同时在原图和增强图上检测
            all_regions = []
            
            # 检测增强图 (对半透明水印更敏感)
            regions_enh, _ = mser.detectRegions(self.gray_enhanced)
            all_regions.extend(regions_enh)
            
            # 检测原图 (补充检测)
            regions_orig, _ = mser.detectRegions(self.gray)
            all_regions.extend(regions_orig)
            
            # 对MSER区域进行分组
            bboxes = []
            for region in all_regions:
                if len(region) < 8:  # 降低阈值以检测更小的字符
                    continue
                    
                x, y, w, h = cv2.boundingRect(region)
                bboxes.append((x, y, x + w, y + h))
            
            if not bboxes:
                return []
            
            # 使用形态学操作合并相邻的文字区域
            mask = np.zeros(self.gray.shape, dtype=np.uint8)
            for x1, y1, x2, y2 in bboxes:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            # 膨胀以连接相邻字符
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 查找连通区域
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # 过滤
                if area < 500 or area > self.width * self.height * 0.5:
                    continue
                
                # 检查区域密度（文字应该有适当的密度）
                roi = self.gray[y:y+h, x:x+w]
                density = np.count_nonzero(roi < 200) / area
                
                if 0.05 < density < 0.8:  # 文字区域的密度范围
                    mser_regions.append({
                        'bbox': (x, y, x + w, y + h),
                        'relative_bbox': (
                            x / self.width,
                            y / self.height,
                            (x + w) / self.width,
                            (y + h) / self.height
                        ),
                        'area': area,
                        'density': density
                    })
        
        except Exception as e:
            print(f"  ⚠️  MSER detection failed: {e}")
        
        return mser_regions
    
    def detect_frequency_patterns(self) -> List[Dict]:
        """
        使用频域分析检测重复性水印模式（优化版）
        对于平铺式的文字水印特别有效
        
        Returns:
            检测到的重复模式区域
        """
        freq_regions = []
        
        try:
            # 1. 使用霍夫变换检测斜线模式
            edges = cv2.Canny(self.gray, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                   minLineLength=50, maxLineGap=30)
            
            if lines is not None and len(lines) > 15:
                # 分析线条角度分布
                angles_list = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angles_list.append(angle)
                
                # 统计角度分布
                angles_array = np.array(angles_list)
                
                # 检测是否有主导角度（表示斜向重复水印）
                hist, bin_edges = np.histogram(angles_array, bins=36, range=(-180, 180))
                dominant_bins = np.where(hist > len(lines) * 0.15)[0]  # 超过15%的线条集中在某个角度
                
                if len(dominant_bins) > 0:
                    print(f"    Detected dominant angle distribution, likely repeated diagonal watermarks")
                    
                    # 不是返回整个区域，而是根据线条分布创建多个区域
                    for dominant_bin in dominant_bins:
                        dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
                        
                        # 找到该角度附近的线条
                        angle_mask = np.abs(angles_array - dominant_angle) < 10
                        relevant_lines = lines[angle_mask]
                        
                        if len(relevant_lines) < 5:
                            continue
                        
                        # 根据线条分布确定水印区域
                        all_points = []
                        for line in relevant_lines:
                            x1, y1, x2, y2 = line[0]
                            all_points.extend([[x1, y1], [x2, y2]])
                        
                        all_points = np.array(all_points)
                        x_min, y_min = all_points.min(axis=0)
                        x_max, y_max = all_points.max(axis=0)
                        
                        # 扩展边界以覆盖完整文字
                        margin = 20
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(self.width, x_max + margin)
                        y_max = min(self.height, y_max + margin)
                        
                        freq_regions.append({
                            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                            'relative_bbox': (
                                x_min / self.width,
                                y_min / self.height,
                                x_max / self.width,
                                y_max / self.height
                            ),
                            'area': (x_max - x_min) * (y_max - y_min),
                            'pattern_type': 'repeated',
                            'angle': dominant_angle,
                            'confidence': len(relevant_lines) / len(lines)
                        })
            
            # 2. FFT频域分析（补充检测）
            dft = cv2.dft(np.float32(self.gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
            magnitude_spectrum = np.log(magnitude_spectrum + 1)
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
            
            # 检测频域峰值
            _, thresh = cv2.threshold(magnitude_spectrum, 245, 255, cv2.THRESH_BINARY)
            center_y, center_x = self.height // 2, self.width // 2
            cv2.circle(thresh, (center_x, center_y), 40, 0, -1)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 如果频域有多个明显峰值，说明有周期性重复
            if len(contours) > 8:
                print(f"    FFT detected {len(contours)} peaks, confirmed repetitive pattern")
                
                # 如果霍夫变换没有检测到，则返回全局区域
                if len(freq_regions) == 0:
                    margin_x = int(self.width * 0.1)
                    margin_y = int(self.height * 0.1)
                    
                    freq_regions.append({
                        'bbox': (margin_x, margin_y, 
                                self.width - margin_x, self.height - margin_y),
                        'relative_bbox': (
                            margin_x / self.width,
                            margin_y / self.height,
                            (self.width - margin_x) / self.width,
                            (self.height - margin_y) / self.height
                        ),
                        'area': (self.width - 2 * margin_x) * (self.height - 2 * margin_y),
                        'pattern_type': 'repeated_global',
                        'confidence': len(contours) / 20
                    })
        
        except Exception as e:
            print(f"  ⚠️  频域分析失败: {e}")
        
        return freq_regions
    
    def detect_bright_regions(self, brightness_threshold: int = 200) -> List[Dict]:
        """
        检测高亮区域（白色水印）
        
        Args:
            brightness_threshold: 亮度阈值
        
        Returns:
            检测到的高亮区域
        """
        # 检测高亮区域
        _, bright_mask = cv2.threshold(self.gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bright_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 过滤
            if area < 500 or area > self.width * self.height * 0.3:
                continue
            
            bright_regions.append({
                'bbox': (x, y, x + w, y + h),
                'relative_bbox': (
                    x / self.width,
                    y / self.height,
                    (x + w) / self.width,
                    (y + h) / self.height
                ),
                'area': area
            })
        
        return bright_regions
    
    def auto_detect(self) -> List[Dict]:
        """
        自动检测水印（综合多种方法）
        支持中文、斜向、重复性等各种水印类型
        
        Returns:
            检测到的所有可能的水印区域
        """
        print("🔍 Starting watermark detection...")
        print("   Supported: Horizontal, Vertical, Diagonal, Chinese, Repeated watermarks")
        
        all_detections = []
        
        # Method 1: Corner detection
        print("\n  → Detecting corner regions...")
        corner_watermarks = self.detect_corner_watermarks()
        for wm in corner_watermarks:
            wm['method'] = 'corner'
            all_detections.append(wm)
        print(f"     Found {len(corner_watermarks)} corner watermarks")
        
        # Method 2: Text region detection (enhanced, multi-directional)
        print("  → Detecting text regions (multi-directional, multi-threshold)...")
        text_regions = self.detect_text_regions()
        for region in text_regions:
            region['method'] = 'text'
            all_detections.append(region)
        print(f"     Found {len(text_regions)} text regions")
        
        # Method 3: Rotated text detection
        print("  → Detecting rotated text (-45° ~ +45°)...")
        rotated_regions = self.detect_rotated_text()
        for region in rotated_regions:
            region['method'] = 'rotated'
            all_detections.append(region)
        print(f"     Found {len(rotated_regions)} rotated text regions")
        
        # Method 4: MSER text detection
        print("  → MSER text detection (character-level)...")
        mser_regions = self.detect_mser_text()
        for region in mser_regions:
            region['method'] = 'mser'
            all_detections.append(region)
        print(f"     Found {len(mser_regions)} MSER text regions")
        
        # Method 5: Frequency analysis (repeated watermarks)
        print("  → Frequency analysis (repeated patterns)...")
        freq_regions = self.detect_frequency_patterns()
        for region in freq_regions:
            region['method'] = 'frequency'
            all_detections.append(region)
        print(f"     Found {len(freq_regions)} repeated pattern regions")
        
        # Method 6: Bright region detection
        print("  → Detecting bright regions...")
        bright_regions = self.detect_bright_regions()
        for region in bright_regions:
            region['method'] = 'bright'
            all_detections.append(region)
        print(f"     Found {len(bright_regions)} bright regions")
        
        # Merge overlapping regions (using aggressive merging strategy)
        print("\n  → Merging overlapping regions...")
        original_count = len(all_detections)
        all_detections = self._merge_overlapping_regions(all_detections, iou_threshold=0.2)  # Lower threshold for easier merging
        print(f"     Merged from {original_count} to {len(all_detections)} regions")
        
        # Second pass filter: Remove too small or too large detections
        filtered_detections = []
        min_area = self.width * self.height * 0.001  # At least 0.1%
        max_area = self.width * self.height * 0.7    # At most 70%
        
        for det in all_detections:
            area = det.get('area', 0)
            if min_area <= area <= max_area:
                filtered_detections.append(det)
            else:
                print(f"     Filtered abnormal region: area={area:.0f} ({area/(self.width*self.height)*100:.1f}%)")
        
        all_detections = filtered_detections
        
        print(f"\n✅ Total detected: {len(all_detections)} possible watermark regions")
        
        return all_detections
    
    def _merge_overlapping_regions(self, regions: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        智能合并重叠的区域（优化版）
        
        Args:
            regions: 区域列表
            iou_threshold: IoU阈值，超过此值的区域会被合并
        
        Returns:
            合并后的区域列表
        """
        if not regions:
            return []
        
        # 先按置信度排序（如果有）
        regions_sorted = sorted(regions, key=lambda x: x.get('confidence', 0.5), reverse=True)
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions_sorted):
            if i in used:
                continue
            
            bbox1 = region1['bbox']
            merged_bbox = list(bbox1)
            merged_methods = [region1.get('method', 'unknown')]
            
            # 查找需要合并的区域
            merge_group = [i]
            
            for j, region2 in enumerate(regions_sorted[i+1:], i+1):
                if j in used:
                    continue
                
                bbox2 = region2['bbox']
                iou = self._calculate_iou(bbox1, bbox2)
                
                # 动态调整阈值：相同方法的区域用更严格的阈值
                method2 = region2.get('method', 'unknown')
                threshold = iou_threshold
                if region1.get('method') == method2:
                    threshold = 0.3  # 相同方法更容易合并
                
                if iou > threshold:
                    merge_group.append(j)
                    merged_bbox[0] = min(merged_bbox[0], bbox2[0])
                    merged_bbox[1] = min(merged_bbox[1], bbox2[1])
                    merged_bbox[2] = max(merged_bbox[2], bbox2[2])
                    merged_bbox[3] = max(merged_bbox[3], bbox2[3])
                    if method2 not in merged_methods:
                        merged_methods.append(method2)
                    used.add(j)
            
            # 检查合并后的区域是否太大（可能是误合并）
            merged_width = merged_bbox[2] - merged_bbox[0]
            merged_height = merged_bbox[3] - merged_bbox[1]
            merged_area = merged_width * merged_height
            
            # 如果合并后区域超过图像50%，可能有问题
            if merged_area < self.width * self.height * 0.7:
                region1['bbox'] = tuple(merged_bbox)
                region1['relative_bbox'] = (
                    merged_bbox[0] / self.width,
                    merged_bbox[1] / self.height,
                    merged_bbox[2] / self.width,
                    merged_bbox[3] / self.height
                )
                
                # 如果合并了多个方法，标注为复合
                if len(merged_methods) > 1:
                    region1['method'] = f"{region1.get('method', 'unknown')}+{len(merged_methods)-1}"
                    region1['merged_from'] = merged_methods
                
                merged.append(region1)
        
        return merged
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _cv2_to_pil(self, cv2_img):
        """OpenCV图像转PIL图像"""
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    
    def _pil_to_cv2(self, pil_img):
        """PIL图像转OpenCV图像"""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _draw_chinese_text(self, img_pil, text, position, font_size=20, color=(255, 255, 255)):
        """
        在PIL图像上绘制中文文字（增强版，支持多种字体fallback）
        
        Args:
            img_pil: PIL图像
            text: 要绘制的文字
            position: (x, y) 坐标
            font_size: 字体大小
            color: RGB颜色元组
        """
        draw = ImageDraw.Draw(img_pil)
        
        # 按优先级尝试多个字体路径
        font_paths = [
            # macOS 中文字体
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            # Linux 中文字体
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            # Windows 中文字体
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\simsun.ttc",
        ]
        
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        # 如果所有字体都失败，使用英文替代显示
        if font is None:
            font = ImageFont.load_default()
            # 中文映射到英文缩写
            text_mapping = {
                '角落': 'Corner',
                '文字': 'Text',
                '高亮': 'Bright',
                '斜向': 'Rotated',
                '重复': 'Repeat',
                '其他': 'Other'
            }
            for cn, en in text_mapping.items():
                text = text.replace(cn, en)
        
        draw.text(position, text, font=font, fill=color)
        return img_pil
    
    def visualize_detections(self, detections: List[Dict], output_path: Path, show_contours: bool = True):
        """
        Visualize detection results with English labels
        
        Args:
            detections: List of detected regions
            output_path: Output image path
            show_contours: Whether to show precise contours (not just bounding boxes)
        """
        vis_img = self.image.copy()
        
        colors = {
            'corner': (0, 0, 255),      # Red
            'text': (0, 255, 0),        # Green
            'bright': (255, 0, 0),      # Blue
            'rotated': (255, 0, 255),   # Magenta
            'mser': (0, 255, 255),      # Yellow
            'frequency': (255, 128, 0), # Orange
            'default': (255, 255, 0)    # Cyan
        }
        
        method_names = {
            'corner': 'Corner',
            'text': 'Text',
            'bright': 'Bright',
            'rotated': 'Rotated',
            'mser': 'MSER',
            'frequency': 'Repeated',
            'default': 'Other'
        }
        
        # Draw contours if requested
        if show_contours:
            for detection in detections:
                x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
                method = detection.get('method', 'default')
                color = colors.get(method, colors['default'])
                
                # Extract ROI and generate precise contours
                if x2 > x1 and y2 > y1:
                    roi = self.gray_enhanced[y1:y2, x1:x2]
                    
                    # Multi-threshold detection
                    _, binary1 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    _, binary2 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    binary = cv2.bitwise_or(binary1, cv2.bitwise_not(binary2))
                    
                    # Find contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Convert contours to original image coordinates
                    for contour in contours:
                        if cv2.contourArea(contour) > 30:  # Filter small noise
                            contour_shifted = contour + np.array([x1, y1])
                            cv2.drawContours(vis_img, [contour_shifted], -1, color, 2)
        
        # Draw bounding boxes and labels
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
            method = detection.get('method', 'default')
            color = colors.get(method, colors['default'])
            method_name = method_names.get(method.split('+')[0], method)  # Handle merged methods
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw corner markers for better visibility
            corner_size = 15
            thickness = 3
            cv2.line(vis_img, (x1, y1), (x1 + corner_size, y1), color, thickness)
            cv2.line(vis_img, (x1, y1), (x1, y1 + corner_size), color, thickness)
            cv2.line(vis_img, (x2, y1), (x2 - corner_size, y1), color, thickness)
            cv2.line(vis_img, (x2, y1), (x2, y1 + corner_size), color, thickness)
            cv2.line(vis_img, (x1, y2), (x1 + corner_size, y2), color, thickness)
            cv2.line(vis_img, (x1, y2), (x1, y2 - corner_size), color, thickness)
            cv2.line(vis_img, (x2, y2), (x2 - corner_size, y2), color, thickness)
            cv2.line(vis_img, (x2, y2), (x2, y2 - corner_size), color, thickness)
            
            # Draw label with background
            if 'angle' in detection:
                label = f"#{idx+1} {method_name} {detection['angle']:.0f}deg"
            else:
                label = f"#{idx+1} {method_name}"
            
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(label_h + 10, y1 - 5)
            
            # Label background
            cv2.rectangle(vis_img, (x1, label_y - label_h - 5), 
                         (x1 + label_w + 10, label_y + 5), color, -1)
            cv2.putText(vis_img, label, (x1 + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw coordinates
            x1_r, y1_r, x2_r, y2_r = detection['relative_bbox']
            coord_text = f"{x1_r:.3f} {y1_r:.3f} {x2_r:.3f} {y2_r:.3f}"
            
            (coord_w, coord_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(vis_img, (x1, y2 + 5), 
                         (x1 + coord_w + 5, y2 + coord_h + 10), (0, 0, 0), -1)
            cv2.putText(vis_img, coord_text, (x1 + 2, y2 + coord_h + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw legend
        legend_x, legend_y = 10, 30
        legend_width = 280
        legend_height = len(colors) * 25 + 40
        
        # Legend background
        cv2.rectangle(vis_img, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (0, 0, 0), -1)
        cv2.rectangle(vis_img, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height),
                     (255, 255, 255), 2)
        
        # Legend title
        cv2.putText(vis_img, "Detection Methods:", (legend_x + 10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend items
        y_offset = legend_y + 45
        for method, color in colors.items():
            method_name = method_names.get(method, method)
            
            # Color box
            cv2.rectangle(vis_img, (legend_x + 10, y_offset - 10), 
                         (legend_x + 25, y_offset + 5), color, -1)
            
            # Method name
            cv2.putText(vis_img, f"{method_name}", 
                       (legend_x + 32, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Add statistics
        stats_y = legend_y + legend_height + 20
        stats_text = f"Total: {len(detections)} regions detected"
        cv2.rectangle(vis_img, (legend_x, stats_y), 
                     (legend_x + legend_width, stats_y + 30), (0, 0, 0), -1)
        cv2.rectangle(vis_img, (legend_x, stats_y), 
                     (legend_x + legend_width, stats_y + 30), (255, 255, 255), 2)
        cv2.putText(vis_img, stats_text, (legend_x + 10, stats_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(str(output_path), vis_img)
        print(f"📷 Visualization saved to: {output_path}")
    
    def extract_watermark_template(
        self,
        detection: Dict,
        output_path: Path,
        padding: int = 5
    ):
        """
        提取水印作为模板
        
        Args:
            detection: 检测到的水印区域
            output_path: 输出模板路径
            padding: 边距
        """
        x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
        
        # 添加边距
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(self.width, x2 + padding)
        y2 = min(self.height, y2 + padding)
        
        # 提取区域
        template = self.image[y1:y2, x1:x2]
        
        cv2.imwrite(str(output_path), template)
        print(f"💾 Watermark template saved to: {output_path}")
    
    def generate_precise_mask(
        self,
        detection: Dict,
        method: str = 'auto',
        dilation_size: int = 5
    ) -> np.ndarray:
        """
        Generate precise mask for detected watermark (following text contours)
        
        Args:
            detection: Detected watermark region
            method: Mask generation method ('auto', 'text_trace', 'bbox', 'contour')
            dilation_size: Dilation kernel size to ensure complete coverage
        
        Returns:
            Binary mask with same size as original image (0=background, 255=watermark)
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width, x2)
        y2 = min(self.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return mask
        
        # Extract ROI
        roi_gray = self.gray_enhanced[y1:y2, x1:x2]
        
        # Auto-select method based on detection type
        if method == 'auto':
            det_method = detection.get('method', 'default').split('+')[0]
            if det_method in ['text', 'rotated', 'mser']:
                method = 'contour'
            else:
                method = 'bbox'
        
        if method == 'bbox':
            # Simple rectangular mask
            mask[y1:y2, x1:x2] = 255
        
        elif method in ['text_trace', 'contour']:
            # Precise contour tracing for text
            roi_mask = self._trace_text_contours(roi_gray)
            mask[y1:y2, x1:x2] = roi_mask
        
        # Dilate to ensure complete coverage
        if dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (dilation_size, dilation_size))
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _trace_text_contours(self, roi_gray: np.ndarray) -> np.ndarray:
        """
        Precisely trace text contours for mask generation
        
        Args:
            roi_gray: Grayscale ROI
        
        Returns:
            ROI mask
        """
        h, w = roi_gray.shape
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Multiple binarization strategies
        binary_methods = []
        
        # 1. Otsu threshold
        _, binary1 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_methods.append(binary1)
        
        # 2. Inverse Otsu (for dark watermarks)
        _, binary2 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_methods.append(binary2)
        
        # 3. Adaptive threshold
        binary3 = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        binary_methods.append(binary3)
        
        # 4. Adaptive inverse
        binary4 = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        binary_methods.append(binary4)
        
        # 5. Multi-level thresholds
        for threshold in [140, 160, 180, 200]:
            _, binary_t = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY)
            binary_methods.append(binary_t)
            _, binary_t_inv = cv2.threshold(roi_gray, threshold, 255, cv2.THRESH_BINARY_INV)
            binary_methods.append(binary_t_inv)
        
        # 6. Edge-based detection
        edges = cv2.Canny(roi_gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        binary_methods.append(edges_dilated)
        
        # Combine all methods
        for binary in binary_methods:
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter small noise
                if area < 20:
                    continue
                
                # Filter too large regions (likely background)
                if area > h * w * 0.9:
                    continue
                
                # Draw filled contour
                cv2.drawContours(combined_mask, [contour], -1, 255, -1)
        
        # Morphological closing to connect broken text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return combined_mask
    
    def generate_and_save_mask(
        self,
        detection: Dict,
        output_path: Path,
        method: str = 'auto',
        visualize: bool = True
    ) -> Path:
        """
        Generate and save precise watermark mask
        
        Args:
            detection: Detected watermark region
            output_path: Mask output path
            method: Mask generation method
            visualize: Whether to generate visualization overlay
        
        Returns:
            Saved mask file path
        """
        # Generate mask
        mask = self.generate_precise_mask(detection, method=method)
        
        # Save mask
        cv2.imwrite(str(output_path), mask)
        print(f"💾 Precise mask saved to: {output_path}")
        
        # Calculate coverage
        coverage = np.count_nonzero(mask) / (self.height * self.width) * 100
        print(f"   Coverage: {coverage:.2f}%")
        
        # Visualization
        if visualize:
            vis_path = output_path.parent / f"{output_path.stem}_preview.jpg"
            self._visualize_mask_overlay(mask, vis_path)
            print(f"📷 Mask preview saved to: {vis_path}")
        
        return output_path
    
    def generate_combined_mask(
        self,
        detections: List[Dict],
        output_path: Path,
        method: str = 'auto',
        visualize: bool = True
    ) -> Path:
        """
        Generate combined mask for multiple detections
        
        Args:
            detections: List of detected watermark regions
            output_path: Mask output path
            method: Mask generation method
            visualize: Whether to generate visualization overlay
        
        Returns:
            Saved mask file path
        """
        # Create empty mask
        combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        print(f"\n🎨 Generating combined mask ({len(detections)} regions)...")
        
        # Generate and combine masks for each detection
        for idx, detection in enumerate(detections):
            print(f"  Processing region #{idx+1}...")
            mask = self.generate_precise_mask(detection, method=method)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Save combined mask
        cv2.imwrite(str(output_path), combined_mask)
        print(f"✅ Combined mask saved to: {output_path}")
        
        # Calculate coverage
        coverage = np.count_nonzero(combined_mask) / (self.height * self.width) * 100
        print(f"   Coverage: {coverage:.2f}%")
        
        # Visualization
        if visualize:
            vis_path = output_path.parent / f"{output_path.stem}_preview.jpg"
            self._visualize_mask_overlay(combined_mask, vis_path)
            print(f"📷 Mask preview saved to: {vis_path}")
        
        return output_path
    
    def _visualize_mask_overlay(self, mask: np.ndarray, output_path: Path):
        """
        Generate mask overlay on original image
        
        Args:
            mask: Binary mask
            output_path: Output path
        """
        # Create colored mask (semi-transparent red)
        overlay = self.image.copy()
        red_mask = np.zeros_like(self.image)
        red_mask[:, :, 2] = mask  # Red channel
        
        # Blend
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1 - alpha, red_mask, alpha, 0)
        
        # Add mask contours (green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        # Add statistics
        mask_pixels = np.count_nonzero(mask)
        total_pixels = self.height * self.width
        coverage = mask_pixels / total_pixels * 100
        
        info_text = [
            f"Mask Coverage: {coverage:.2f}%",
            f"Mask Pixels: {mask_pixels:,}",
            f"Image Size: {self.width}x{self.height}"
        ]
        
        # Draw info box
        y_offset = 30
        for text in info_text:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (10, y_offset - text_h - 5), 
                         (20 + text_w, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(overlay, text, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += text_h + 15
        
        cv2.imwrite(str(output_path), overlay)
    
    def batch_generate_masks(
        self,
        detections: List[Dict],
        output_dir: Path,
        method: str = 'auto',
        visualize: bool = False
    ) -> List[Path]:
        """
        Batch generate individual masks for each detection
        
        Args:
            detections: List of detected watermark regions
            output_dir: Mask output directory
            method: Mask generation method
            visualize: Whether to generate visualization overlays
        
        Returns:
            List of generated mask file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_paths = []
        print(f"\n🎨 Batch generating masks ({len(detections)} regions)...")
        
        for idx, detection in enumerate(detections):
            det_method = detection.get('method', 'unknown').split('+')[0]
            print(f"  Region #{idx+1} ({det_method})...")
            
            # Generate filename
            mask_path = output_dir / f"{self.image_path.stem}_mask_{idx+1}_{det_method}.png"
            
            # Generate and save mask
            self.generate_and_save_mask(detection, mask_path, method=method, visualize=visualize)
            mask_paths.append(mask_path)
        
        print(f"\n✅ Generated {len(mask_paths)} mask files")
        
        return mask_paths


def interactive_selection(image_path: Path) -> Optional[Dict]:
    """
    交互式选择水印区域（使用OpenCV GUI）
    
    Args:
        image_path: 图片路径
    
    Returns:
        选择的区域信息
    """
    print("\n🖱️  交互式选择模式")
    print("   使用鼠标框选水印区域，按 'c' 确认，按 'r' 重新选择，按 'q' 退出")
    
    img = cv2.imread(str(image_path))
    if img is None:
        print("错误：无法读取图片")
        return None
    
    height, width = img.shape[:2]
    
    # 调整窗口大小
    max_width = 1200
    if width > max_width:
        scale = max_width / width
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
        height, width = img.shape[:2]
    
    clone = img.copy()
    roi = None
    selecting = False
    start_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_point, roi, img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_point = (x, y)
            img = clone.copy()
        
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            img = clone.copy()
            cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            roi = (start_point[0], start_point[1], x, y)
            cv2.rectangle(img, start_point, (x, y), (0, 255, 0), 2)
    
    cv2.namedWindow('选择水印区域')
    cv2.setMouseCallback('选择水印区域', mouse_callback)
    
    while True:
        cv2.imshow('选择水印区域', img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and roi:  # 确认
            cv2.destroyAllWindows()
            x1, y1, x2, y2 = roi
            # 确保坐标正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 恢复到原始尺寸的坐标
            if width > max_width:
                scale = width / max_width
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            
            return {
                'bbox': (x1, y1, x2, y2),
                'relative_bbox': (
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height
                ),
                'method': 'interactive'
            }
        
        elif key == ord('r'):  # 重新选择
            img = clone.copy()
            roi = None
        
        elif key == ord('q'):  # 退出
            cv2.destroyAllWindows()
            return None
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="水印自动检测和提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect watermarks
  python detect_watermark.py -i sample.jpg
  
  # Auto-detect and visualize
  python detect_watermark.py -i sample.jpg --visualize
  
  # Auto-detect and generate mask (RECOMMENDED)
  python detect_watermark.py -i sample.jpg --generate-mask --mask-preview
  
  # Interactive selection and generate mask
  python detect_watermark.py -i sample.jpg --interactive --generate-mask --mask-preview
  
  # Generate separate masks for each detected region
  python detect_watermark.py -i sample.jpg --generate-mask --separate-masks --mask-output ./masks
  
  # Use different mask generation methods
  python detect_watermark.py -i sample.jpg --generate-mask --mask-method contour
  
  # Extract watermark as template
  python detect_watermark.py -i sample.jpg --extract 0 --template watermark.png
  
  # Save detection results as config file
  python detect_watermark.py -i sample.jpg --save-config watermark_config.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='输入图片路径'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='可视化检测结果'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='交互式选择水印区域'
    )
    
    parser.add_argument(
        '--extract',
        type=int,
        help='提取指定索引的水印作为模板（从0开始）'
    )
    
    parser.add_argument(
        '--template',
        type=Path,
        default=Path('watermark_template.png'),
        help='水印模板输出路径'
    )
    
    parser.add_argument(
        '--save-config',
        type=Path,
        help='保存检测结果为配置文件'
    )
    
    parser.add_argument(
        '--generate-mask',
        action='store_true',
        help='Generate precise watermark mask (following text contours)'
    )
    
    parser.add_argument(
        '--mask-output',
        type=Path,
        help='Mask output path (default: same directory as input with _mask.png suffix)'
    )
    
    parser.add_argument(
        '--mask-method',
        type=str,
        choices=['auto', 'text_trace', 'contour', 'bbox'],
        default='auto',
        help='Mask generation method: auto (automatic), text_trace/contour (precise contours), bbox (simple rectangle)'
    )
    
    parser.add_argument(
        '--separate-masks',
        action='store_true',
        help='Generate separate mask file for each detected region'
    )
    
    parser.add_argument(
        '--mask-preview',
        action='store_true',
        help='Generate mask preview (overlay on original image)'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.input.exists():
        print(f"错误：文件不存在: {args.input}")
        return
    
    print("=" * 60)
    print("水印自动检测工具")
    print("=" * 60)
    print(f"\n输入图片: {args.input}")
    
    if args.interactive:
        # 交互式选择
        detection = interactive_selection(args.input)
        if detection is None:
            print("❌ 未选择区域")
            return
        
        detections = [detection]
        print(f"\n✅ 选择的区域: {detection['relative_bbox']}")
    else:
        # 自动检测
        detector = WatermarkDetector(args.input)
        detections = detector.auto_detect()
        
        if not detections:
            print("\n❌ 未检测到水印")
            print("\n💡 建议:")
            print("   1. 使用交互式模式: --interactive")
            print("   2. 手动指定区域参数")
            return
    
    # 显示检测结果
    print("\n" + "=" * 60)
    print("检测结果:")
    print("=" * 60)
    for idx, det in enumerate(detections):
        x1_r, y1_r, x2_r, y2_r = det['relative_bbox']
        method = det.get('method', 'unknown')
        print(f"\n区域 #{idx+1} (方法: {method})")
        print(f"  相对坐标: {x1_r:.4f} {y1_r:.4f} {x2_r:.4f} {y2_r:.4f}")
        print(f"  命令参数: --region {x1_r:.4f} {y1_r:.4f} {x2_r:.4f} {y2_r:.4f}")
    
    # 可视化
    if args.visualize or args.interactive:
        detector = WatermarkDetector(args.input)
        vis_path = args.input.parent / f"{args.input.stem}_detected.jpg"
        detector.visualize_detections(detections, vis_path)
    
    # 提取模板
    if args.extract is not None:
        if 0 <= args.extract < len(detections):
            detector = WatermarkDetector(args.input)
            detector.extract_watermark_template(
                detections[args.extract],
                args.template
            )
            
            print(f"\n📝 使用此模板批量处理:")
            print(f"   python generate_masks.py -i ./images -o ./masks \\")
            print(f"     --template {args.template}")
        else:
            print(f"\n❌ 错误: 索引 {args.extract} 超出范围 (0-{len(detections)-1})")
    
    # 保存配置
    if args.save_config:
        config = {
            'image': str(args.input),
            'detections': [
                {
                    'relative_bbox': det['relative_bbox'],
                    'method': det.get('method', 'unknown')
                }
                for det in detections
            ]
        }
        
        with open(args.save_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Config saved to: {args.save_config}")
    
    # Generate mask
    if args.generate_mask:
        detector = WatermarkDetector(args.input)
        
        if args.separate_masks:
            # Generate separate mask for each region
            output_dir = args.mask_output or args.input.parent / 'masks'
            mask_paths = detector.batch_generate_masks(
                detections,
                output_dir,
                method=args.mask_method,
                visualize=args.mask_preview
            )
            
            print(f"\n📝 Use these masks to remove watermarks:")
            print(f"   export KMP_DUPLICATE_LIB_OK=TRUE")
            print(f"   # Process each mask separately or merge them")
            
        else:
            # Generate combined mask
            if args.mask_output:
                mask_path = args.mask_output
            else:
                mask_path = args.input.parent / f"{args.input.stem}_mask.png"
            
            detector.generate_combined_mask(
                detections,
                mask_path,
                method=args.mask_method,
                visualize=args.mask_preview
            )
            
            print(f"\n📝 Use this mask to remove watermarks:")
            print(f"   export KMP_DUPLICATE_LIB_OK=TRUE")
            print(f"   iopaint run --model=lama --device=cpu \\")
            print(f"     --image={args.input} --mask={mask_path} --output=./output")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    
    if not args.generate_mask:
        print("\n💡 Generate precise watermark mask (RECOMMENDED):")
        print(f"\n   python detect_watermark.py -i {args.input} \\")
        print(f"     --generate-mask --mask-preview --mask-method contour")
    
    print("\n1. Use detected coordinates to batch generate masks:")
    if detections:
        regions_args = ' '.join([
            f"--region {det['relative_bbox'][0]:.4f} {det['relative_bbox'][1]:.4f} "
            f"{det['relative_bbox'][2]:.4f} {det['relative_bbox'][3]:.4f}"
            for det in detections
        ])
        print(f"\n   python generate_masks.py -i ./images -o ./masks \\")
        print(f"     {regions_args}")
    
    print("\n2. Or use template matching:")
    if args.extract is not None:
        print(f"\n   python generate_masks.py -i ./images -o ./masks \\")
        print(f"     --template {args.template}")
    
    print("\n3. Batch remove watermarks:")
    print("\n   export KMP_DUPLICATE_LIB_OK=TRUE")
    print("   iopaint run --model=lama --device=cpu \\")
    print("     --image=./images --mask=./masks --output=./output")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

