"""
优化版水印检测器 - 专注于传统方法的精简优化
============================================

核心理念：传统方法优化，保护原图质量

问题分析 - 当前方法的局限性：
1. 流程复杂：OCR检测 → 过滤 → 轮廓提取 → 形态学操作 → 主体保护 → 桥接闭运算
2. 补救措施堆积：为了解决误伤，引入无数启发式规则，难以维护
3. AI方法破坏原图：大模型推理会模糊整个图像，破坏像素质量
4. 缺乏针对性：没有专门优化传统方法的检测效果

优化方案：多策略传统检测 + 轻量级处理 + 保护原图
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import easyocr
from PIL import Image


@dataclass
class WatermarkDetection:
    """水印检测结果"""
    bbox: np.ndarray  # 边界框
    confidence: float  # 置信度
    text: str  # 识别的文字
    mask: np.ndarray  # 精确mask
    category: str  # 分类：watermark/text/subject


class OptimizedWatermarkDetector:
    """
    优化版水印检测器 - 专注于传统方法优化

    核心创新：
    1. 多策略检测：多种边缘检测并行，提高召回率
    2. 保护原图：轻量级处理，避免像素破坏
    3. 智能过滤：基于几何特征过滤误检
    4. 精确mask：生成高质量的二值mask
    """

    def __init__(self, enable_preview: bool = True):
        # 专注优化传统方法，保护原图质量
        self.enable_preview = enable_preview

        # 轻量级OCR - 只用于验证，不用于主要检测
        try:
            self.reader = easyocr.Reader(['en', 'ch_sim'], gpu=torch.cuda.is_available())
        except:
            self.reader = None

    def detect_watermarks(self, image: np.ndarray, preview_path: Optional[str] = None) -> List[WatermarkDetection]:
        """
        主检测流程 - 专注于传统方法的优化

        阶段1: 多策略边缘检测 - 识别潜在水印区域（保护原图）
        阶段2: 特征验证 - 确认水印特征
        阶段3: 精确mask生成 - 生成最终mask
        """
        print("🚀 Starting traditional watermark detection...")

        # 初始化预览图像（不修改原图）
        preview_image = image.copy() if self.enable_preview else None

        # 阶段1: 多策略传统检测（保护原图质量）
        candidate_regions = self._traditional_localization(image)
        print(f"📍 Traditional method located {len(candidate_regions)} candidate regions")

        # 生成阶段1预览
        if preview_image is not None and candidate_regions:
            self._draw_detection_preview(preview_image, candidate_regions, stage=1,
                                       title="Stage 1: Candidates", color=(255, 255, 0))

        # 阶段2: 特征验证和过滤
        valid_watermarks = self._feature_verification(image, candidate_regions)
        print(f"✅ Validation passed {len(valid_watermarks)} watermark regions")

        # 生成阶段2预览
        if preview_image is not None and valid_watermarks:
            valid_regions = [info['bbox'] for info in valid_watermarks]
            self._draw_detection_preview(preview_image, valid_regions, stage=2,
                                       title="Stage 2: Validated", color=(0, 255, 255))

        # 阶段3: 精确mask生成
        detections = self._generate_precise_masks(image, valid_watermarks)
        print(f"🎯 Generated {len(detections)} precise watermark masks")

        # 生成最终预览
        if preview_image is not None and detections:
            final_regions = [det.bbox for det in detections]
            self._draw_detection_preview(preview_image, final_regions, stage=3,
                                       title="Stage 3: Final", color=(0, 255, 0))

            # 保存预览图像
            if preview_path:
                self._save_detection_preview(preview_image, detections, preview_path)

        return detections

    def _traditional_localization(self, image: np.ndarray) -> List[np.ndarray]:
        """
        阶段1: 多策略传统检测 - 保护原图质量

        核心优化：
        - 使用轻量级边缘检测，避免模糊
        - 多尺度并行检测，提高召回率
        - 智能合并，减少误检
        - 保护原图像素，不进行破坏性处理
        """
        print("🔍 Using multi-strategy traditional detection...")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_candidates = []

        # 策略1: 轻量级Canny边缘检测（保护细节）
        edges1 = cv2.Canny(gray, 30, 80)  # 更低的阈值，检测更细的边缘
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 更小的核
        closed1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel1, iterations=1)
        candidates1 = self._extract_regions_from_mask(closed1, min_area=15)  # 更小的最小面积
        all_candidates.extend(candidates1)

        # 策略2: 基于对比度的检测（检测半透明水印）
        blur = cv2.GaussianBlur(gray, (3, 3), 0)  # 轻微模糊保护细节
        contrast = cv2.absdiff(gray, blur)
        _, thresh2 = cv2.threshold(contrast, 8, 255, cv2.THRESH_BINARY)  # 更低的阈值
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel2, iterations=1)
        candidates2 = self._extract_regions_from_mask(closed2, min_area=12)
        all_candidates.extend(candidates2)

        # 策略3: 自适应阈值检测（适应不同亮度）
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed3 = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel3, iterations=1)
        candidates3 = self._extract_regions_from_mask(closed3, min_area=10)
        all_candidates.extend(candidates3)

        # 智能去重合并
        all_candidates = self._merge_overlapping_regions(all_candidates, iou_threshold=0.5)

        # 过滤明显不是水印的区域（基于形状特征）
        filtered_candidates = []
        for region in all_candidates:
            x1, y1, x2, y2 = region
            w, h = x2 - x1, y2 - y1

            # 过滤过大或过小的区域 (放宽限制)
            area = w * h
            if area < 10 or area > image.shape[0] * image.shape[1] * 0.15:  # 允许更大的区域，降低最小面积
                continue

            # 过滤宽高比异常的区域 (放宽限制)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
            if aspect_ratio > 30:  # 放宽宽高比限制
                continue

            filtered_candidates.append(region)

        print(f"🎯 Multi-strategy detection found {len(filtered_candidates)} regions")
        return filtered_candidates

    def _feature_verification(self, image: np.ndarray, candidate_regions: List[np.ndarray]) -> List[Dict]:
        """
        阶段2: 多特征验证

        验证标准：
        1. 重复性：水印通常重复出现
        2. 透明度：水印通常半透明
        3. 位置：水印通常在图像边缘或角落
        4. 一致性：相似区域应该有相似特征
        """
        verified_regions = []

        for region in candidate_regions:
            features = self._extract_watermark_features(image, region)

            # 综合评分
            score = self._compute_watermark_score(features)

            if score > 0.15:  # 进一步降低阈值，确保不遗漏水印
                verified_regions.append({
                    'bbox': region,
                    'features': features,
                    'score': score
                })

        return verified_regions

    def _generate_precise_masks(self, image: np.ndarray, verified_regions: List[Dict]) -> List[WatermarkDetection]:
        """
        阶段3: 生成精确mask

        方法：
        1. 基于AI分割结果
        2. 结合局部对比度分析
        3. 形态学优化
        """
        detections = []

        for region_info in verified_regions:
            bbox = region_info['bbox']

            # 扩大边界框以包含标点符号
            expanded_bbox = self._expand_bbox_for_punctuation(image, bbox)

            # 使用扩大后的边界框生成mask，确保标点也被包含
            precise_mask = self._refine_mask_with_contrast(image, expanded_bbox)

            # OCR验证（可选，用于提取文字内容）
            text = ""
            confidence = region_info['score']

            if self.reader:
                try:
                    roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    results = self.reader.readtext(roi, detail=0)
                    if results:
                        text = results[0]
                except:
                    pass

            detection = WatermarkDetection(
                bbox=bbox,
                confidence=confidence,
                text=text,
                mask=precise_mask,
                category='watermark'
            )

            detections.append(detection)

        return detections

    def _expand_bbox_for_punctuation(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        智能扩大文字边界框，确保包含标点符号

        策略：
        1. 向右下方扩展，覆盖可能的标点位置
        2. 基于文字尺寸计算合适的扩展范围
        3. 避免扩展到图像边界外
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # 计算文字尺寸
        text_width = x2 - x1
        text_height = y2 - y1

        # 扩展策略：
        # 右边：扩展文字宽度的30-50%，覆盖句号、逗号等
        # 下边：扩展文字高度的20-40%，覆盖下标点
        # 左边：轻微扩展，避免遗漏
        # 上边：轻微扩展

        expand_right = int(text_width * 0.4)   # 向右扩展40%
        expand_bottom = int(text_height * 0.3) # 向下扩展30%
        expand_left = int(text_width * 0.1)    # 向左扩展10%
        expand_top = int(text_height * 0.1)    # 向上扩展10%

        # 应用扩展，但不超出图像边界
        new_x1 = max(0, x1 - expand_left)
        new_y1 = max(0, y1 - expand_top)
        new_x2 = min(w, x2 + expand_right)
        new_y2 = min(h, y2 + expand_bottom)

        return np.array([new_x1, new_y1, new_x2, new_y2])

    def _extract_watermark_features(self, image: np.ndarray, bbox: np.ndarray) -> Dict:
        """提取水印特征"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        features = {}

        # 1. 透明度特征（水印通常半透明）
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        features['transparency'] = self._compute_transparency_score(gray)

        # 2. 重复性特征
        features['repetitiveness'] = self._compute_repetition_score(roi)

        # 3. 位置特征（水印常在边缘）
        features['position'] = self._compute_position_score(bbox, image.shape)

        # 4. 对比度特征（水印对比度适中）
        features['contrast'] = self._compute_contrast_score(roi)

        return features

    def _compute_watermark_score(self, features: Dict) -> float:
        """计算综合水印评分"""
        weights = {
            'transparency': 0.3,
            'repetitiveness': 0.3,
            'position': 0.2,
            'contrast': 0.2
        }

        score = sum(features[key] * weights.get(key, 0) for key in features.keys())
        return min(1.0, max(0.0, score))

    def _compute_transparency_score(self, gray_roi: np.ndarray) -> float:
        """计算透明度评分（水印通常半透明，不太暗也不太亮）"""
        hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        # 水印通常在中间灰度范围
        mid_range = hist[64:192].sum()
        return float(mid_range)

    def _compute_repetition_score(self, roi: np.ndarray) -> float:
        """计算重复性评分"""
        # 简化的重复性检测：基于FFT频域分析
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 计算FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # 重复图案通常在频域有明显峰值
        # 这里简化为计算频域能量分布的均匀性
        hist, _ = np.histogram(magnitude_spectrum.flatten(), bins=50)
        hist = hist / hist.sum()

        # 均匀分布说明有重复图案
        uniformity = 1.0 - np.std(hist)
        return float(uniformity)

    def _compute_position_score(self, bbox: np.ndarray, image_shape: Tuple) -> float:
        """计算位置评分（水印常在边缘）"""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算到图像中心的距离
        dist_to_center = np.sqrt(((center_x - w/2) / (w/2)) ** 2 +
                                ((center_y - h/2) / (h/2)) ** 2)

        # 水印通常不在图像中心（距离中心越远越可能是水印）
        return min(1.0, dist_to_center)

    def _compute_contrast_score(self, roi: np.ndarray) -> float:
        """计算对比度评分（水印对比度适中）"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 计算局部对比度
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(gray, kernel)
        eroded = cv2.erode(gray, kernel)

        contrast = cv2.absdiff(dilated, eroded)
        mean_contrast = np.mean(contrast)

        # 水印对比度通常适中（不高不低）
        normalized_contrast = min(1.0, mean_contrast / 50.0)
        return normalized_contrast

    def _refine_mask_with_contrast(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """基于多尺度对比度分析精确化mask - 优化版，确保文字和标点都被检测"""
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        # 预处理：保持细节的双边滤波
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 使用较保守的双边滤波参数，保持更多细节
        bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # 优化的对比度检测 - 确保文字和标点都被检测
        # 使用两种尺度的对比度检测：主要文字和细节标点

        # 主要文字检测（中等尺度）
        blur_main = cv2.GaussianBlur(bilateral, (3, 3), 0)
        contrast_main = cv2.absdiff(bilateral, blur_main)

        # 标点细节检测（小尺度，更敏感）
        blur_detail = cv2.GaussianBlur(bilateral, (1, 1), 0)
        contrast_detail = cv2.absdiff(bilateral, blur_detail)

        # 结合两种对比度：主要文字 + 标点细节
        combined_contrast = cv2.addWeighted(contrast_main, 0.7, contrast_detail, 0.3, 0)

        # 优化的二值化策略 - 确保文字和标点都被正确分割
        # 使用自适应阈值 + OTSU，确保细节被保留
        thresh_adaptive = cv2.adaptiveThreshold(
            combined_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2
        )

        # OTSU阈值作为补充，确保弱对比度区域也被检测
        _, thresh_otsu = cv2.threshold(combined_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 合并两种阈值结果，确保文字和标点都被覆盖
        combined_thresh = cv2.bitwise_or(thresh_adaptive, thresh_otsu)

        # 优化的形态学处理 - 连接文字和标点，确保完整性
        # 步骤1: 中等闭运算连接断开的笔画和标点
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        # 步骤2: 小幅膨胀确保覆盖所有水印细节
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        refined = cv2.dilate(refined, kernel_small, iterations=1)

        # 步骤3: 最终闭运算确保标点与文字完全连接
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

        # 步骤4: Canny边缘补充 - 检测可能遗漏的标点边缘
        canny = cv2.Canny(bilateral, 20, 60)  # 适中的阈值，避免噪声
        # 只在对比度区域添加Canny边缘
        _, contrast_mask = cv2.threshold(combined_contrast, 10, 255, cv2.THRESH_BINARY)
        canny_filtered = cv2.bitwise_and(canny, contrast_mask)
        refined = cv2.bitwise_or(refined, canny_filtered)

        # 步骤5: 轻微清理 - 去除孤立噪声点
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        # 创建全尺寸mask
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = refined

        return full_mask

    def _extract_regions_from_mask(self, mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
        """从mask提取区域bbox"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append(np.array([x, y, x + w, y + h]))

        return regions

    def _merge_overlapping_regions(self, regions: List[np.ndarray], iou_threshold: float = 0.3) -> List[np.ndarray]:
        """合并重叠的候选区域，减少重复"""
        if not regions:
            return regions

        merged = []
        used = [False] * len(regions)

        for i, region1 in enumerate(regions):
            if used[i]:
                continue

            x1_1, y1_1, x2_1, y2_1 = region1
            merged_region = region1.copy()

            for j, region2 in enumerate(regions):
                if i == j or used[j]:
                    continue

                x1_2, y1_2, x2_2, y2_2 = region2

                # 计算交并比 (IoU)
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)

                if x2_i > x1_i and y2_i > y1_i:
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > iou_threshold:
                        # 合并区域
                        merged_region[0] = min(merged_region[0], region2[0])  # x1
                        merged_region[1] = min(merged_region[1], region2[1])  # y1
                        merged_region[2] = max(merged_region[2], region2[2])  # x2
                        merged_region[3] = max(merged_region[3], region2[3])  # y2
                        used[j] = True

            merged.append(merged_region)

        return merged


    def _draw_detection_preview(self, image: np.ndarray, regions: List[np.ndarray],
                               stage: int, title: str, color: Tuple[int, int, int]):
        """在预览图像上绘制检测结果"""
        if not self.enable_preview:
            return

        # 绘制边框
        for region in regions:
            if len(region) == 4:  # [x1, y1, x2, y2] 格式
                x1, y1, x2, y2 = region
            else:  # 多边形格式
                x1, y1 = region.min(axis=0)
                x2, y2 = region.max(axis=0)

            # 绘制矩形边框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 添加区域编号
            cv2.putText(image, str(len(regions)), (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 添加阶段标题 (英文避免乱码)
        h, w = image.shape[:2]
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        cv2.putText(image, f"{title} ({len(regions)} regions)", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _save_detection_preview(self, image: np.ndarray, detections: List[WatermarkDetection],
                               output_path: str):
        """保存检测预览图像"""
        if not self.enable_preview:
            return

        # 创建最终预览
        final_preview = image.copy()

        # 在右上角添加统计信息
        h, w = final_preview.shape[:2]
        stats_overlay = final_preview.copy()
        cv2.rectangle(stats_overlay, (w - 350, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(stats_overlay, 0.7, final_preview, 0.3, 0, final_preview)

        # 计算统计信息
        total_area = sum((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]) for det in detections)
        coverage = total_area / (h * w) * 100

        # 添加统计文本 (英文避免乱码)
        cv2.putText(final_preview, f"Detection Stats", (w - 340, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(final_preview, f"Regions: {len(detections)}", (w - 340, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(final_preview, f"Coverage: {coverage:.1f}%", (w - 340, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(final_preview, f"Method: Traditional", (w - 340, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 保存到对应的轮次目录
        import os
        preview_dir = os.path.dirname(output_path)
        os.makedirs(preview_dir, exist_ok=True)

        preview_name = os.path.basename(output_path).replace('.png', '_detection_preview.jpg')
        preview_path = os.path.join(preview_dir, preview_name)

        cv2.imwrite(preview_path, final_preview)
        print(f"📸 Detection preview saved: {preview_path}")

    def generate_mask(self, image: np.ndarray, preview_path: Optional[str] = None) -> np.ndarray:
        """
        生成最终水印mask - 简化的主接口

        返回: 二值mask (255=水印区域, 0=背景)
        """
        detections = self.detect_watermarks(image, preview_path)

        # 合并所有检测结果
        final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for detection in detections:
            final_mask = cv2.bitwise_or(final_mask, detection.mask)

        # 最终形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        coverage = np.count_nonzero(final_mask) / final_mask.size * 100
        print(f"💾 Mask coverage: {coverage:.1f}%")
        return final_mask


def main():
    """使用示例 - 支持轮次目录结构"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description='优化版水印检测器')
    parser.add_argument('-r', '--round', required=True, help='轮次目录 (如: 1, 2, 3)')
    parser.add_argument('--preview', action='store_true', help='生成检测过程预览图')
    parser.add_argument('--simple-preview', action='store_true', help='生成简单的最终结果预览图')
    parser.add_argument('--no-preview', action='store_true', help='禁用所有预览功能')

    args = parser.parse_args()

    # 构建路径
    round_dir = args.round
    input_path = 'sample.jpg'
    output_path = os.path.join(round_dir, 'mask.png')

    # 确保轮次目录存在
    os.makedirs(round_dir, exist_ok=True)

    # 加载图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Failed to load image: {input_path}")
        return

    print(f"🎯 Processing round {args.round}: {input_path}")

    # 创建检测器
    enable_preview = not args.no_preview
    detector = OptimizedWatermarkDetector(enable_preview=enable_preview)

    # 确定预览路径
    preview_path = None
    if args.preview or args.simple_preview:
        preview_path = os.path.join(round_dir, 'detection_preview.jpg')

    # 生成mask
    mask = detector.generate_mask(image, preview_path)

    # 保存结果
    cv2.imwrite(output_path, mask)
    print(f"💾 Mask saved: {output_path}")

    # 生成简单预览（如果需要且还没有生成检测预览）
    if args.simple_preview and not args.preview:
        simple_preview_path = os.path.join(round_dir, 'simple_preview.jpg')
        overlay = image.copy()
        overlay[mask > 127] = [0, 0, 255]  # 红色标记水印区域

        # 添加简单的统计信息
        h, w = overlay.shape[:2]
        coverage = cv2.countNonZero(mask) / (h * w) * 100
        cv2.putText(overlay, f"Coverage: {coverage:.1f}%", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(overlay, f"Round: {args.round}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(simple_preview_path, overlay)
        print(f"🖼️ Simple preview saved: {simple_preview_path}")

    # 自动运行水印去除
    print("🧹 Starting automatic watermark removal...")
    run_watermark_removal(round_dir, input_path, output_path)

    print(f"✅ Round {args.round} completed!")

def run_watermark_removal(round_dir: str, input_image: str, mask_file: str):
    """运行水印去除命令"""
    import subprocess
    import os

    # 首先检查iopaint是否可用
    try:
        result = subprocess.run(["iopaint", "--help"], capture_output=True, text=True, timeout=10)
        iopaint_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        iopaint_available = False

    if not iopaint_available:
        print("⚠️ iopaint command not found. Skipping automatic watermark removal.")
        print("💡 To enable automatic removal, install IOPaint and ensure it's in PATH")
        print("   Manual command format:")
        print(f"   iopaint run --model=lama --device=cpu --image={input_image} --mask={mask_file} --output={round_dir}/output.jpg")
        return

    # 构建输出路径（在同一目录下）
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_file = round_dir

    # 构建iopaint命令
    cmd = [
        "iopaint", "run",
        "--model=lama",
        "--device=cpu",
        f"--image={input_image}",
        f"--mask={mask_file}",
        f"--output={output_file}"
    ]

    print(f"🔧 Running: {' '.join(cmd)}")

    try:
        # 设置环境变量避免库冲突
        env = os.environ.copy()
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # 执行命令
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✨ Watermark removal completed: {output_file}")

            # 检查输出文件是否存在
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"📊 Output file size: {file_size} bytes")
            else:
                print("⚠️ Output file was not created")

        else:
            print(f"❌ Watermark removal failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error details: {result.stderr[:500]}...")
            print("💡 Manual command:")
            print(f"   export KMP_DUPLICATE_LIB_OK=TRUE")
            print(f"   {' '.join(cmd)}")

    except subprocess.TimeoutExpired:
        print("⏰ Watermark removal timed out (5 minutes)")
        print("💡 Try running manually with shorter timeout or different model")
    except Exception as e:
        print(f"❌ Unexpected error during watermark removal: {str(e)}")
        print("💡 Manual command:")
        print(f"   export KMP_DUPLICATE_LIB_OK=TRUE")
        print(f"   {' '.join(cmd)}")

if __name__ == "__main__":
    main()
