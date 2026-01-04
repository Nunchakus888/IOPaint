#!/usr/bin/env python3
"""
OCR-Based Watermark Detection
Uses EasyOCR to detect text regions (including diagonal Chinese text)
Most accurate method for text watermarks

========================
中文说明（核心原理概览）
========================
本脚本的目标：生成一个二值 mask（白=需要修复/去除的水印区域，黑=保留原图），用于 iopaint/lama 等修复模型。

整体思路分两步：
1) “找哪里有水印文字”(detection)：用 OCR（EasyOCR）在多种增强版本的图上尽可能高召回地检测文本框 boxes；
2) “框内画得更像文字而不是整块矩形”(masking)：在每个 box 内做笔画/轮廓级提取，生成更贴字的 mask，减少误伤主体。

关键对象：
- box / boxes：
  - 一个 box 表示 OCR 检出的单个文字区域四边形框；
  - 数据结构：np.ndarray，形状通常为 (4, 1, 2) 或 (4, 2)；每个点是 (x, y)；
  - boxes 是 box 的列表：List[np.ndarray]。

- mask：
  - 二值图（uint8），形状为 (H, W)；
  - 约定：0=背景(保留)，255=需要修复(去水印)。

为什么要“高召回 boxes + 框内精细 mask”？
- 仅靠 box 填充矩形（rect）覆盖全面但容易过大误伤，且文字笔画间隙/抗锯齿边缘容易残留；
- 仅靠精细轮廓提取容易漏掉一些 faint 水印，需要先把候选区域尽量找全；
- 因此使用：raw 高召回检测（text regions 330+） + 框内轮廓/笔画mask +（可选）方向约束/主体保护后处理。
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCRWatermarkDetector:
    """Detect watermarks using OCR text detection"""
    
    def __init__(self, image_path: Path):
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR not installed. Install with:\n"
                "  pip install easyocr"
            )
        
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        self.height, self.width = self.image.shape[:2]
        
        print("🔧 Initializing EasyOCR (first time may download models)...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        print("✅ EasyOCR ready")
    
    def _normalize_text(self, s: str) -> str:
        """Normalize OCR text for clustering (remove spaces/punct, lower).

        中文：用于把 OCR 输出文本做归一化，便于统计“重复水印文案”。
        水印通常会重复出现相同的短语/字符（例如某个 App 名、网站名等）。
        """
        if not s:
            return ""
        s = s.strip().lower()
        # keep letters/numbers/CJK, drop most punctuation/spaces
        out = []
        for ch in s:
            if ch.isspace():
                continue
            # CJK Unified Ideographs
            if "\u4e00" <= ch <= "\u9fff":
                out.append(ch)
                continue
            if ch.isalnum():
                out.append(ch)
                continue
        return "".join(out)

    def _box_angle(self, box: np.ndarray) -> float:
        """
        Estimate box angle in degrees, normalized to [-45, 45].
        Box shape can be (4,1,2) or (4,2).
        """
        # 中文：水印通常有一个全局统一的倾斜角（例如 -20° 左右）。
        # 利用 box 的角度可以：
        # - 在 mask 提取时做“方向约束”：不符合主角度的巨大连通域更可能是人物轮廓/背景结构；
        # - 在 rect 模式下做“沿水印方向的桥接闭运算”：补齐字符间隙，减少残留。
        pts = box.reshape(-1, 2)
        p0, p1 = pts[0], pts[1]
        angle = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90
        return angle

    def _compute_dominant_angle(self, boxes: List[np.ndarray]) -> Optional[float]:
        """Compute dominant angle (deg) across boxes; returns None if not enough boxes."""
        if not boxes or len(boxes) < 8:
            return None
        angles = np.array([self._box_angle(b) for b in boxes], dtype=np.float32)
        bin_edges = np.arange(-45, 50, 5, dtype=np.float32)
        hist, _ = np.histogram(angles, bins=bin_edges)
        dom_bin = int(np.argmax(hist))
        dom_center = float((bin_edges[dom_bin] + bin_edges[dom_bin + 1]) / 2.0)
        return dom_center

    def _rotate_mask(self, mask: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate a binary mask around image center; keep same size."""
        h, w = mask.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rotated

    def _close_along_angle(self, mask: np.ndarray, angle_deg: Optional[float], k_long: int, k_short: int) -> np.ndarray:
        """
        桥接闭运算 (Bridging Closing Operation) - 解决文字间隙残留问题

        🎯 问题背景:
        - rect模式下检出率高(43%+)但存在"文字间隙的残留"
        - 字符间细小间隙未被桥接，修复后仍有水印痕迹

        🔧 核心原理：方向感知的形态学闭运算
        - 水印文字有一致倾斜角度(45度)，利用此特性进行精准桥接
        - 普通闭运算=膨胀+腐蚀，填补间隙但方向随机
        - 角度感知闭运算：只沿着水印方向连接，避免误伤垂直元素

        📏 参数机制:
        - k_long: 沿着文字延伸方向的长轴(9-45像素)，覆盖字符间距
        - k_short: 垂直文字方向的短轴(3-11像素)，避免过度扩张
        - 计算方式: k_long = mw*0.30, k_short = mh*0.10 (mw/mh为文字框平均尺寸)

        🔄 旋转策略:
        1. 逆旋转使水印变水平: rot = rotate(mask, -angle_deg)
        2. 水平方向闭运算: morphologyEx(rot, MORPH_CLOSE, (k_long, k_short))
        3. 旋转回原角度: rotate(result, angle_deg)

        ⚡ 优势:
        - 精准桥接：只沿水印方向连接
        - 自适应尺寸：根据文字实际尺寸调整
        - 保持紧致：k_short较小避免垂直扩张
        - 角度无关：无论水印角度都能正确桥接
        """
        # 如果没有角度信息，使用标准闭运算（适用于水平文字）
        if angle_deg is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_long, k_short))
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 🎨 角度感知桥接闭运算流程

        # 步骤1: 逆旋转使水印笔画变水平，便于沿文字方向桥接
        # 例如：45度水印 -> 旋转-45度 -> 变为水平，便于水平方向闭运算
        rot = self._rotate_mask(mask, -angle_deg)

        # 步骤2: 创建长条形核进行水平方向闭运算
        # k_long(9-45): 沿着文字延伸方向的长距离，覆盖字符间距
        # k_short(3-11): 垂直文字方向的短距离，避免垂直过度扩张
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_long, k_short))

        # 执行形态学闭运算：填补k_long范围内的水平间隙
        # 闭运算 = 先膨胀再腐蚀，连接相近的笔画区域
        rot = cv2.morphologyEx(rot, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 步骤3: 旋转回原始角度，恢复水印的原始方向
        out = self._rotate_mask(rot, angle_deg)

        # 步骤4: 二值化处理，确保输出为干净的二值mask
        _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
        return out

    def _remove_subject_components(self, mask: np.ndarray, dom_angle: Optional[float]) -> np.ndarray:
        """
        Remove subject-like components from a high-recall mask while keeping watermark strokes.

        Strategy (tuned for repeated diagonal watermark on photos):
        - Compute gradient magnitude on the original image (subject edges are strong).
        - For each connected component in mask:
          remove if it's large, mostly vertical, centered, strong-edge, and NOT aligned to watermark dominant angle.

        中文解释（为什么这样能“减少人物误伤”且不丢掉人物身上的水印）：
        - raw 高召回检测会把一些“人物轮廓/衣服边界”也画进 mask；
        - 但人物身上的水印也是真的需要去除，不能整块把人物区域抹掉；
        - 因此这里只删除“强梯度(强边缘)”像素：人物轮廓通常梯度很强，
          水印笔画更偏“中等/弱梯度”（半透明），会尽量被保留下来。
        """
        if mask is None or mask.size == 0:
            return mask

        h, w = mask.shape[:2]
        img_area = float(h * w)
        if img_area <= 0:
            return mask

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        # normalize to 0..255 for stable thresholds
        grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
        grad_thr = float(np.percentile(grad_norm, 75))  # "strong edge" baseline

        clean = mask.copy()

        # Erode a bit to break thin connections (watermark strokes) from large subject blobs
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        work = cv2.erode(mask, k, iterations=1)
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # thresholds
        large_area = img_area * 0.0035  # ~0.35% of image
        very_large_area = img_area * 0.008  # ~0.8% of image

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < large_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            aspect = float(bh) / float(bw)  # vertical if > 1
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            centered = (w * 0.22 < cx < w * 0.78) and (h * 0.12 < cy < h * 0.92)
            fill_ratio = area / max(1.0, float(bw * bh))

            # orientation via PCA major axis
            pts = cnt.reshape(-1, 2).astype(np.float32)
            mean = pts.mean(axis=0, keepdims=True)
            cov = np.cov((pts - mean).T)
            try:
                vals, vecs = np.linalg.eig(cov)
                major = vecs[:, int(np.argmax(vals))]
                ang = float(np.degrees(np.arctan2(major[1], major[0])))
            except Exception:
                ang = 0.0
            while ang > 45:
                ang -= 90
            while ang < -45:
                ang += 90

            # strong-edge score within component
            comp_mask = np.zeros_like(mask)
            cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
            # map back to original (pre-erode) by dilating the component slightly
            comp_mask = cv2.dilate(comp_mask, k, iterations=1)
            gvals = grad_norm[comp_mask > 0]
            gmean = float(np.mean(gvals)) if gvals.size else 0.0

            # decide removal
            # Subject-like: large + vertical-ish + centered + strong gradient
            subj_like = centered and (aspect > 1.15) and (gmean > grad_thr) and (fill_ratio > 0.25)
            if not subj_like and area < very_large_area:
                continue

            if dom_angle is not None:
                if abs(ang - float(dom_angle)) <= 22:
                    # aligned with watermark direction, keep
                    continue

            # If extremely large or subject-like: only remove strong-edge pixels inside the region,
            # keep low-contrast watermark strokes (which tend to have lower gradients).
            if area >= very_large_area or subj_like:
                remove_mask = np.zeros_like(mask)
                remove_mask[(comp_mask > 0) & (grad_norm > grad_thr)] = 255
                clean[remove_mask > 0] = 0

        return clean

    def detect_text_boxes_raw(self, low_text: float = 0.3) -> List[np.ndarray]:
        """
        High-recall detection (no filtering): main + enhanced + edge + complex-zone,
        then de-dup by center distance (keeps ~300+ regions on repeated watermarks).

        中文解释（raw 高召回检测为什么能到 330+）：
        - main：原图直接 OCR；
        - recall pass：把 low_text/text_threshold/link_threshold 放宽再跑一次，补 faint 水印；
        - enhanced：对比度增强/反色/二值化/L通道/gamma 多路增强再 OCR，提高不同背景下召回；
        - edge：pad 后检测边缘文字；
        - complex：只在“已发现区域附近 + 复杂背景区域”再跑低阈值 OCR 做补漏；
        - 最后用中心距离去重，避免重复框爆炸。

        输出：boxes（List[np.ndarray]），每个 box 是 OCR 四边形框，用于后续画 mask。
        """
        print("🔍 Detecting text regions (raw, high-recall)...")

        all_boxes: List[np.ndarray] = []

        # Main + recall pass
        dets1 = self._readtext_to_detections(
            self.image,
            low_text=low_text,
            text_threshold=0.5,
            link_threshold=0.3,
            width_ths=0.7,
        )
        recall_low_text = max(0.10, low_text - 0.10)
        dets2 = self._readtext_to_detections(
            self.image,
            low_text=recall_low_text,
            text_threshold=0.40,
            link_threshold=0.25,
            width_ths=0.7,
        )
        all_boxes.extend([d["box"] for d in dets1])
        all_boxes.extend([d["box"] for d in dets2])
        print(f"  Main: {len(dets1)} regions")

        # Enhanced
        enh = self._detect_on_enhanced_detections()
        all_boxes.extend([d["box"] for d in enh])

        # Edge
        ed = self._detect_edge_text_detections()
        all_boxes.extend([d["box"] for d in ed])

        # Complex (near existing)
        if all_boxes:
            dets_c = self._detect_missed_watermarks_detections(all_boxes)
            all_boxes.extend([d["box"] for d in dets_c])
        
        # De-dupe by center distance (legacy behavior that yields ~330 regions)
        all_boxes = self._remove_duplicate_boxes(all_boxes)
        
        # Store dominant watermark angle for later contour filtering
        self._dominant_watermark_angle = self._compute_dominant_angle(all_boxes)
        if self._dominant_watermark_angle is not None:
            print(f"  Dominant angle: {self._dominant_watermark_angle:.1f}°")
        
        print(f"✅ Total: {len(all_boxes)} text regions")
        return all_boxes
    
    def _readtext_to_detections(self, image_bgr: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Run EasyOCR and return detections with box/text/conf.
        Each detection: {box: np.ndarray(4,1,2), text: str, conf: float}
        """
        # 中文：
        # EasyOCR 的 readtext 返回 (box, text, conf)：
        # - box：四点坐标（四边形），表示检测到的文字区域；
        # - text：识别出的字符串（可能有误识别/空）；
        # - conf：置信度。
        # 这里封装为 dict，后续既能用 box 做定位，又能用 text/conf 做过滤与统计。
        dets: List[Dict[str, Any]] = []
        results = self.reader.readtext(image_bgr, **kwargs)
        for det in results:
            if not det or len(det) < 2:
                continue
            box = np.array(det[0], dtype=np.int32).reshape((-1, 1, 2))
            text = det[1] if len(det) > 1 else ""
            conf = float(det[2]) if len(det) > 2 else 0.0
            dets.append({"box": box, "text": text, "conf": conf})
        return dets

    def _dedupe_detections(self, dets: List[Dict[str, Any]], dist_thresh: float = 20.0) -> List[Dict[str, Any]]:
        """Deduplicate detections by center distance; keep higher confidence / longer text."""
        if len(dets) < 2:
            return dets
        centers = [d["box"].mean(axis=0)[0] for d in dets]
        used = set()
        out: List[Dict[str, Any]] = []
        for i, (d, c) in enumerate(zip(dets, centers)):
            if i in used:
                continue
            best = d
            used.add(i)
            for j in range(i + 1, len(dets)):
                if j in used:
                    continue
                if np.linalg.norm(c - centers[j]) < dist_thresh:
                    other = dets[j]
                    used.add(j)
                    # choose best by confidence, then by normalized text length
                    if other.get("conf", 0.0) > best.get("conf", 0.0) + 1e-6:
                        best = other
                    elif abs(other.get("conf", 0.0) - best.get("conf", 0.0)) <= 1e-6:
                        if len(self._normalize_text(other.get("text", ""))) > len(self._normalize_text(best.get("text", ""))):
                            best = other
            out.append(best)
        return out

    def _filter_watermark_detections(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter OCR detections to likely watermark text using:
        - repeated normalized text tokens (watermarks repeat)
        - dominant rotation angle cluster
        Keeps recall by allowing medium confidence if text repeats.
        """
        if not dets:
            return dets

        # Count repeated texts
        counts: Dict[str, float] = {}
        for d in dets:
            nt = self._normalize_text(d.get("text", ""))
            conf = float(d.get("conf", 0.0))
            if len(nt) < 2:
                continue
            if conf < 0.15:
                continue
            counts[nt] = counts.get(nt, 0.0) + (0.5 + conf)

        # pick top repeated tokens
        top_tokens = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        token_set = {t for t, score in top_tokens if score >= 2.0}  # be more recall-friendly

        # dominant angle (watermark is usually consistent)
        angles = np.array([self._box_angle(d["box"]) for d in dets], dtype=np.float32)
        # histogram bins of 5 degrees in [-45,45]
        bin_edges = np.arange(-45, 50, 5, dtype=np.float32)
        hist, _ = np.histogram(angles, bins=bin_edges)
        dom_bin = int(np.argmax(hist))
        dom_center = float((bin_edges[dom_bin] + bin_edges[dom_bin + 1]) / 2.0)

        def angle_ok(a: float) -> bool:
            # allow +-12 degrees, plus also allow the symmetric due to 90deg normalization jitter
            return abs(a - dom_center) <= 12.0

        # adaptive size filter (reject extreme false positives while keeping recall)
        areas = []
        for d in dets:
            try:
                areas.append(float(cv2.contourArea(d["box"].reshape(-1, 2))))
            except Exception:
                continue
        med_area = float(np.median(areas)) if areas else 0.0
        min_area = max(20.0, med_area * 0.15) if med_area > 0 else 20.0
        max_area = med_area * 6.0 if med_area > 0 else float("inf")

        filtered: List[Dict[str, Any]] = []
        for d in dets:
            nt = self._normalize_text(d.get("text", ""))
            conf = float(d.get("conf", 0.0))
            a = self._box_angle(d["box"])
            try:
                area = float(cv2.contourArea(d["box"].reshape(-1, 2)))
            except Exception:
                area = med_area

            if area < min_area or area > max_area:
                continue

            # Hard reject: looks like OCR noise (too short) and low confidence
            if len(nt) < 2 and conf < 0.60:
                continue

            # keep if text repeats (strong watermark prior)
            if nt and nt in token_set:
                filtered.append(d)
                continue

            # keep if aligned with dominant angle and at least some confidence
            if conf >= 0.15 and angle_ok(a):
                # For low confidence & non-repeated, verify quickly to reduce non-text false positives.
                # This is slower, but only triggered for the ambiguous tail.
                if conf < 0.30:
                    if not self._verify_text_region(d["box"]):
                        continue
                filtered.append(d)
                continue

            # ultra high confidence: keep
            if conf >= 0.80:
                filtered.append(d)
        # if filter is too aggressive, fallback to original
        if len(filtered) < max(20, int(len(dets) * 0.40)):
            return dets
        return filtered

    def detect_text_detections(self, low_text: float = 0.3, enable_template: bool = False) -> List[Dict[str, Any]]:
        """
        Detect text with OCR and keep metadata (box/text/conf).
        This is the new core API; detect_text_regions() remains for compatibility.
        """
        print("🔍 Detecting text regions...")
        all_dets: List[Dict[str, Any]] = []

        # 1) Main detection (two-pass for higher recall)
        main_dets = self._readtext_to_detections(
            self.image,
            low_text=low_text,
            text_threshold=0.5,
            link_threshold=0.3,
            width_ths=0.7,
        )
        # extra recall pass (more permissive)
        recall_low_text = max(0.10, low_text - 0.10)
        main_dets2 = self._readtext_to_detections(
            self.image,
            low_text=recall_low_text,
            text_threshold=0.40,
            link_threshold=0.25,
            width_ths=0.7,
        )
        all_dets.extend(main_dets)
        all_dets.extend(main_dets2)
        print(f"  Main: {len(main_dets)} regions")

        # 2) Enhanced detections
        all_dets.extend(self._detect_on_enhanced_detections())

        # 3) Edge detections
        all_dets.extend(self._detect_edge_text_detections())

        # 4) Targeted detection in complex background near existing
        if all_dets:
            all_dets.extend(self._detect_missed_watermarks_detections([d["box"] for d in all_dets]))

        # 5) Dedupe
        all_dets = self._dedupe_detections(all_dets)

        # 6) Filter to likely watermark text (avoid false positives)
        filtered = self._filter_watermark_detections(all_dets)

        # 7) Template matching is disabled by default (often unstable / low ROI for this dataset)
        if enable_template:
            try:
                tpl_boxes = self._detect_by_template_matching([d["box"] for d in filtered])
                for b in tpl_boxes:
                    filtered.append({"box": b, "text": "", "conf": 0.0})
            except Exception as e:
                print(f"  ⚠️ Template matching skipped: {e}")

        filtered = self._dedupe_detections(filtered)
        print(f"✅ Total: {len(filtered)} text regions")
        return filtered

    def detect_text_regions(self, low_text: float = 0.3) -> List[np.ndarray]:
        """
        Detect all text regions including overlapping areas
        
        Args:
            low_text: Lower threshold for text detection
        """
        # Keep original behavior: return high-recall boxes (matches the 330+ target)
        return self.detect_text_boxes_raw(low_text=low_text)
    
    def _detect_on_enhanced_detections(self) -> List[Dict[str, Any]]:
        """Detect watermarks on enhanced images (for different backgrounds)"""
        dets: List[Dict[str, Any]] = []
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: CLAHE enhanced (for low contrast areas)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        dets.extend(self._readtext_to_detections(enhanced_bgr, low_text=0.25))
        
        # Method 2: Inverted (light watermarks on dark backgrounds)
        inverted = cv2.bitwise_not(self.image)
        dets.extend(self._readtext_to_detections(inverted, low_text=0.3))
        
        # Method 3: High contrast (black/white boundary areas)
        _, high_contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hc_bgr = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
        dets.extend(self._readtext_to_detections(hc_bgr, low_text=0.2))
        
        # Method 4: LAB L-channel (for blue-black boundary)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        l_enhanced = clahe.apply(l_channel)
        l_bgr = cv2.cvtColor(l_enhanced, cv2.COLOR_GRAY2BGR)
        dets.extend(self._readtext_to_detections(l_bgr, low_text=0.25))
        
        # Method 5: Gamma correction (for white backgrounds)
        gamma = 0.5  # Darken to reveal light watermarks
        gamma_corrected = np.power(gray / 255.0, gamma) * 255
        gamma_bgr = cv2.cvtColor(gamma_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dets.extend(self._readtext_to_detections(gamma_bgr, low_text=0.25))
        
        print(f"  Enhanced: {len(dets)} regions")
        return dets
    
    def _remove_duplicate_boxes(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """Remove overlapping duplicate boxes"""
        if len(boxes) < 2:
            return boxes
        
        # Calculate centers
        centers = []
        for box in boxes:
            center = box.mean(axis=0)[0]
            centers.append(center)
        
        # Remove duplicates (boxes with centers close together)
        unique_boxes = []
        used = set()
        
        for i, (box, center) in enumerate(zip(boxes, centers)):
            if i in used:
                continue
            
            # Check if similar to any previous box
            is_duplicate = False
            for j in range(i):
                if j in used:
                    continue
                dist = np.linalg.norm(center - centers[j])
                if dist < 20:  # Too close, likely duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_boxes.append(box)
            used.add(i)
        
        return unique_boxes
    
    def _detect_edge_text_detections(self) -> List[Dict[str, Any]]:
        """Detect text near edges by padding image"""
        pad = 50  # Padding size
        h, w = self.height, self.width
        
        # Pad image
        padded = cv2.copyMakeBorder(
            self.image, pad, pad, pad, pad,
            cv2.BORDER_REFLECT
        )
        
        # Detect on padded image
        results = self.reader.readtext(
            padded,
            low_text=0.25,  # Even lower threshold for edges
            text_threshold=0.4
        )
        
        dets: List[Dict[str, Any]] = []
        for detection in results:
            if not detection or len(detection) < 2:
                continue
            box = np.array(detection[0], dtype=np.int32)
            text = detection[1] if len(detection) > 1 else ""
            conf = float(detection[2]) if len(detection) > 2 else 0.0
            # Adjust coordinates back to original
            box = box - pad
            
            # Keep only boxes that touch edges
            x_min, y_min = box.min(axis=0)
            x_max, y_max = box.max(axis=0)
            
            touches_edge = (x_min < 20 or y_min < 20 or 
                           x_max > w - 20 or y_max > h - 20)
            
            # Clip to image bounds
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            
            if touches_edge and cv2.contourArea(box) > 100:
                dets.append({"box": box.reshape((-1, 1, 2)), "text": text, "conf": conf})
        
        print(f"  Edge detection: {len(dets)} regions")
        return dets
    
    def _detect_by_template_matching(self, existing_boxes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Use detected watermarks as templates to find similar repeated patterns.
        This is most effective for repeated text watermarks; non-text will be filtered.
        """
        if len(existing_boxes) == 0:
            return []
        
        print("🔍 Template matching (using detected watermarks as templates)...")
        new_boxes = []
        all_matches = []  # Collect all matches before filtering
        
        # Extract templates from detected watermarks
        templates = self._extract_templates(existing_boxes)
        if not templates:
            return []
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        # Search for each template
        # for template_idx, template_info in enumerate(templates):
        #     angle = template_info['angle']
        #     size = template_info['size']
            
        #     # Multi-feature template matching (structure + color)
        #     matches = self._match_template_multi_feature(
        #         gray, lab, template_info, angle, size
        #     )
        #     print(f"    Template {template_idx+1}: angle={angle:.2f}, size={size}, raw matches={len(matches)}")
            
        #     # Collect all matches with metadata
        #     for match in matches:
        #         match['template_idx'] = template_idx
        #         all_matches.append(match)
        
        # print(f"    Total raw matches: {len(all_matches)}")
        
        # Sort by similarity (best first)
        all_matches.sort(key=lambda m: m['similarity'], reverse=True)
        
        # Process matches with adaptive verification
        accepted = 0
        skipped_low_sim = 0
        skipped_not_text = 0
        skipped_text_mismatch = 0
        for match in all_matches:
            box = match['box']
            similarity = match['similarity']
            
            # Lower threshold for better recall
            if similarity < 0.33:
                skipped_low_sim += 1
                continue
            
            # Always enforce text-likeness first
            if not self._is_text_like(box):
                skipped_not_text += 1
                continue

            # OCR check (lower threshold to catch faint text)
            ocr_ok = self._verify_text_region(box)

            # If template has text, enforce text match
            tpl = templates[match.get("template_idx", 0)] if templates else {}
            tpl_text = tpl.get("text")
            if tpl_text:
                if not self._ocr_text_match(box, tpl_text):
                    skipped_text_mismatch += 1
                    continue
                # If text matches, accept with relaxed similarity (recall-first)
                if similarity >= 0.40 or ocr_ok:
                    new_boxes.append(box)
                    accepted += 1
                else:
                    skipped_low_sim += 1
            else:
                # No template text available; require both similarity and OCR
                if similarity >= 0.50 and ocr_ok:
                    new_boxes.append(box)
                    accepted += 1
                elif similarity >= 0.60:  # high similarity, allow if text-like
                    new_boxes.append(box)
                    accepted += 1
                else:
                    skipped_not_text += 1
        
        print(f"    Accepted: {accepted}, skipped_low_sim: {skipped_low_sim}, skipped_not_text: {skipped_not_text}, skipped_text_mismatch: {skipped_text_mismatch}")
        
        # Remove duplicates with existing boxes
        new_boxes = self._filter_duplicates_with_existing(new_boxes, existing_boxes)
        
        print(f"  Template matching: {len(new_boxes)} new regions (after filtering)")
        return new_boxes
    
    def _extract_templates(self, boxes: List[np.ndarray]) -> List[dict]:
        """
        Extract template features from detected watermark boxes.
        Focus on text structure and color, not background.
        """
        templates = []
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        
        for box in boxes:
            box_2d = box.reshape(-1, 2)
            x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
            x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
            
            if x_max <= x_min + 10 or y_max <= y_min + 10:
                continue
            
            # Extract ROI with padding
            pad = 5
            x_min_pad = max(0, x_min - pad)
            y_min_pad = max(0, y_min - pad)
            x_max_pad = min(self.width, x_max + pad)
            y_max_pad = min(self.height, y_max + pad)
            
            roi_gray = gray[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            roi_bgr = self.image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            roi_lab = lab[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            
            # Calculate angle (for rotated watermarks)
            p0, p1 = box_2d[0], box_2d[1]
            angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0]) * 180 / np.pi
            while angle > 45:
                angle -= 90
            while angle < -45:
                angle += 90
            
            h, w = roi_gray.shape
            if h < 15 or w < 15 or h > 200 or w > 500:
                continue
            
            # Extract watermark features (text structure, not background)
            features = self._extract_watermark_features(roi_gray, roi_bgr, roi_lab)

            # OCR text content for template (to enforce same text in matches)
            tpl_text = None
            try:
                ocr_res = self.reader.readtext(roi_bgr, low_text=0.2, text_threshold=0.3)
                if ocr_res:
                    # pick the longest detected text
                    ocr_res.sort(key=lambda r: len(r[1]) if len(r) > 1 else 0, reverse=True)
                    tpl_text = ocr_res[0][1].strip()
                    if tpl_text == "":
                        tpl_text = None
            except Exception:
                tpl_text = None
            
            templates.append({
                'template_gray': roi_gray,  # Original for basic matching
                'template_edge': features['edge'],  # Edge features (structure)
                'template_text': features['text_mask'],  # Text mask (structure)
                'template_color': features['color'],  # Color features
                'angle': angle,
                'size': (w, h),
                'original_box': box,
                'color_stats': features['color_stats'],  # Color statistics
                'text': tpl_text
            })
        
        # Use more templates for better coverage (up to 8, prefer medium-sized)
        if len(templates) > 8:
            templates.sort(key=lambda t: abs(t['size'][0] * t['size'][1] - 5000))
            templates = templates[:8]
        
        print(f"  Extracted {len(templates)} templates with features")
        return templates
    
    def _extract_watermark_features(self, roi_gray: np.ndarray, roi_bgr: np.ndarray, 
                                    roi_lab: np.ndarray) -> dict:
        """Extract watermark-specific features: text structure and color"""
        features = {}
        
        # 1. Edge features (text structure, background-independent)
        edges = cv2.Canny(roi_gray, 50, 150)
        features['edge'] = edges
        
        # 2. Text mask (extract text pixels, ignore background)
        # Use local contrast to separate text from background
        local_mean = cv2.blur(roi_gray, (15, 15))
        diff = cv2.absdiff(roi_gray, local_mean)
        _, text_mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        
        # Combine with edge information
        text_mask = cv2.bitwise_or(text_mask, edges)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        features['text_mask'] = text_mask
        
        # 3. Color features (watermark color, not background)
        # Extract color in LAB space (more perceptually uniform)
        l, a, b = cv2.split(roi_lab)
        
        # Mask out background (use text mask)
        text_mask_bool = text_mask > 128
        if np.any(text_mask_bool):
            # Get watermark color statistics
            l_values = l[text_mask_bool]
            a_values = a[text_mask_bool]
            b_values = b[text_mask_bool]
            
            color_stats = {
                'l_mean': float(np.mean(l_values)),
                'l_std': float(np.std(l_values)),
                'a_mean': float(np.mean(a_values)),
                'a_std': float(np.std(a_values)),
                'b_mean': float(np.mean(b_values)),
                'b_std': float(np.std(b_values))
            }
        else:
            # Fallback: use overall statistics
            color_stats = {
                'l_mean': float(np.mean(l)),
                'l_std': float(np.std(l)),
                'a_mean': float(np.mean(a)),
                'a_std': float(np.std(a)),
                'b_mean': float(np.mean(b)),
                'b_std': float(np.std(b))
            }
        
        features['color_stats'] = color_stats
        
        # Create color template (normalized watermark color)
        # Use LAB L channel (brightness) for color matching
        features['color'] = l
        
        return features
    
    def _match_template_multi_feature(self, image_gray: np.ndarray, image_lab: np.ndarray,
                                     template_info: dict, angle: float, 
                                     original_size: Tuple[int, int]) -> List[dict]:
        """
        Match template using multiple features (edge, text structure, color).
        Background-independent matching.
        """
        matches = []
        
        # Use edge template for structure matching (most background-independent)
        template_edge = template_info['template_edge']
        template_text = template_info['template_text']
        template_color = template_info['template_color']
        color_stats = template_info['color_stats']
        
        h, w = template_edge.shape
        
        # Scale factors
        scales = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
        
        # Rotation angles
        angles_to_try = [0]
        if abs(angle) > 5:
            angles_to_try = [angle - 5, angle, angle + 5]
        
        # Extract edge features from full image
        image_edges = cv2.Canny(image_gray, 50, 150)
        image_l = image_lab[:, :, 0]  # LAB L channel for color matching
        
        for scale in scales:
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w < 10 or new_h < 10 or new_w > image_gray.shape[1] or new_h > image_gray.shape[0]:
                continue
            
            # Scale templates
            scaled_edge = cv2.resize(template_edge, (new_w, new_h))
            scaled_text = cv2.resize(template_text, (new_w, new_h))
            scaled_color = cv2.resize(template_color, (new_w, new_h))
            
            for rot_angle in angles_to_try:
                # Rotate templates if needed
                if abs(rot_angle) > 1:
                    center = (new_w // 2, new_h // 2)
                    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    new_w_rot = int((new_h * sin) + (new_w * cos))
                    new_h_rot = int((new_h * cos) + (new_w * sin))
                    M[0, 2] += (new_w_rot / 2) - center[0]
                    M[1, 2] += (new_h_rot / 2) - center[1]
                    
                    rotated_edge = cv2.warpAffine(scaled_edge, M, (new_w_rot, new_h_rot), borderValue=0)
                    rotated_text = cv2.warpAffine(scaled_text, M, (new_w_rot, new_h_rot), borderValue=0)
                    rotated_color = cv2.warpAffine(scaled_color, M, (new_w_rot, new_h_rot), borderValue=128)
                else:
                    rotated_edge = scaled_edge
                    rotated_text = scaled_text
                    rotated_color = scaled_color
                    new_w_rot, new_h_rot = new_w, new_h
                
                if rotated_edge.shape[0] > image_gray.shape[0] or rotated_edge.shape[1] > image_gray.shape[1]:
                    continue
                
                # Multi-feature matching
                # 1. Edge structure matching (most important, background-independent)
                result_edge = cv2.matchTemplate(image_edges, rotated_edge, cv2.TM_CCOEFF_NORMED)
                
                # 2. Text structure matching
                result_text = cv2.matchTemplate(image_edges, rotated_text, cv2.TM_CCOEFF_NORMED)
                
                # 3. Color matching (using LAB L channel)
                result_color = cv2.matchTemplate(image_l, rotated_color, cv2.TM_CCOEFF_NORMED)
                
                # Combine similarities (weighted)
                # Edge is most important (structure), then text, then color
                combined_result = (result_edge * 0.5 + result_text * 0.3 + result_color * 0.2)
                
                # Find matches
                # Lower threshold to improve recall; later stages (text-like + OCR) filter false positives.
                threshold = 0.30
                locations = np.where(combined_result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    x, y = pt
                    similarity = float(combined_result[y, x])
                    
                    # Additional color verification at match location
                    roi_l = image_l[y:y+new_h_rot, x:x+new_w_rot]
                    if roi_l.size > 0:
                        color_similarity = self._match_color_features(
                            roi_l, color_stats
                        )
                        # Boost similarity if color matches
                        if color_similarity > 0.6:
                            similarity = min(1.0, similarity * 1.1)
                    
                    # Create box
                    expand = 3
                    box = np.array([
                        [max(0, x - expand), max(0, y - expand)],
                        [min(image_gray.shape[1], x + new_w_rot + expand), max(0, y - expand)],
                        [min(image_gray.shape[1], x + new_w_rot + expand), min(image_gray.shape[0], y + new_h_rot + expand)],
                        [max(0, x - expand), min(image_gray.shape[0], y + new_h_rot + expand)]
                    ], dtype=np.int32).reshape((-1, 1, 2))
                    
                    matches.append({
                        'box': box,
                        'similarity': similarity,
                        'scale': scale,
                        'angle': rot_angle
                    })
        
        # Remove overlapping matches
        if len(matches) > 0:
            matches = self._filter_overlapping_matches(matches)
        
        return matches
    
    def _match_color_features(self, roi_l: np.ndarray, color_stats: dict) -> float:
        """Match color features (LAB L channel statistics)"""
        if roi_l.size == 0:
            return 0.0
        
        l_mean = np.mean(roi_l)
        l_std = np.std(roi_l)
        
        # Compare with template color stats
        l_diff = abs(l_mean - color_stats['l_mean']) / 255.0
        l_std_diff = abs(l_std - color_stats['l_std']) / 255.0
        
        # Similarity based on how close the statistics are
        similarity = 1.0 - min(1.0, (l_diff + l_std_diff) / 2.0)
        
        return similarity

    def _ocr_text_match(self, box: np.ndarray, tpl_text: str) -> bool:
        """
        Check if OCR text inside the box matches template text.
        Accepts partial matches to handle low-contrast text.
        """
        tpl_text_norm = tpl_text.strip().lower()
        if not tpl_text_norm:
            return False
        
        box_2d = box.reshape(-1, 2)
        x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
        x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        roi = self.image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return False
        
        try:
            results = self.reader.readtext(roi, low_text=0.2, text_threshold=0.3)
            for det in results:
                if len(det) < 2:
                    continue
                text = det[1].strip().lower()
                if not text:
                    continue
                # partial match allowed
                if tpl_text_norm in text or text in tpl_text_norm:
                    return True
        except Exception:
            return False
        
        return False
    
    def _filter_overlapping_matches(self, matches: List[dict]) -> List[dict]:
        """Filter overlapping matches, keep highest similarity"""
        if len(matches) <= 1:
            return matches
        
        # Sort by similarity (descending)
        matches.sort(key=lambda m: m['similarity'], reverse=True)
        
        filtered = []
        used = set()
        
        for i, match in enumerate(matches):
            if i in used:
                continue
            
            box1 = match['box']
            box1_2d = box1.reshape(-1, 2)
            x1_min, y1_min = box1_2d.min(axis=0)
            x1_max, y1_max = box1_2d.max(axis=0)
            center1 = box1.mean(axis=0)[0]
            
            filtered.append(match)
            
            # Mark overlapping matches as used
            for j, other in enumerate(matches[i+1:], start=i+1):
                if j in used:
                    continue
                
                box2 = other['box']
                box2_2d = box2.reshape(-1, 2)
                x2_min, y2_min = box2_2d.min(axis=0)
                x2_max, y2_max = box2_2d.max(axis=0)
                center2 = box2.mean(axis=0)[0]
                
                # Check overlap using IoU-like metric
                # Calculate intersection
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)
                
                if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                    union_area = box1_area + box2_area - inter_area
                    
                    # Only filter if significant overlap (IoU > 0.5)
                    if union_area > 0 and inter_area / union_area > 0.5:
                        used.add(j)
                else:
                    # No intersection, check distance (very close = duplicate)
                    dist = np.linalg.norm(center1 - center2)
                    box1_size = min(x1_max - x1_min, y1_max - y1_min)
                    if dist < box1_size * 0.3:  # Very close, likely duplicate
                        used.add(j)
        
        return filtered
    
    def _verify_text_region(self, box: np.ndarray) -> bool:
        """Verify that matched region contains text using OCR"""
        box_2d = box.reshape(-1, 2)
        x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
        x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        # Extract ROI with padding
        pad = 5
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(self.width, x_max + pad)
        y_max = min(self.height, y_max + pad)
        
        roi = self.image[y_min:y_max, x_min:x_max]
        
        # Quick OCR check (low threshold to catch faint text)
        try:
            results = self.reader.readtext(roi, low_text=0.2, text_threshold=0.3)
            # If OCR finds text, it's likely a watermark
            return len(results) > 0
        except:
            return False

    def _is_text_like(self, box: np.ndarray) -> bool:
        """
        Heuristic to ensure the region looks like text, avoiding non-text elements.
        Uses edge density, aspect ratio, and contrast-based text mask.
        """
        box_2d = box.reshape(-1, 2)
        x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
        x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        roi = self.image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return False
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = roi_gray.shape
        if h < 8 or w < 8:
            return False
        
        # Aspect ratio check (text not extremely tall or wide)
        aspect = max(h, w) / max(1, min(h, w))
        if aspect > 12:  # Extremely elongated shapes are unlikely text
            return False
        
        # Edge density
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if edge_ratio < 0.01:  # Too few edges, likely non-text
            return False
        
        # Local contrast text mask
        local_mean = cv2.blur(roi_gray, (9, 9))
        diff = cv2.absdiff(roi_gray, local_mean)
        _, text_mask = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)
        text_coverage = np.count_nonzero(text_mask) / text_mask.size
        
        # Text coverage should be moderate (not filling entire box)
        if text_coverage < 0.005 or text_coverage > 0.6:
            return False
        
        return True
    
    def _filter_duplicates_with_existing(self, new_boxes: List[np.ndarray], 
                                         existing_boxes: List[np.ndarray]) -> List[np.ndarray]:
        """Filter new boxes that overlap with existing ones"""
        if not existing_boxes:
            return new_boxes
        
        filtered = []
        existing_centers = [box.mean(axis=0)[0] for box in existing_boxes]
        
        for new_box in new_boxes:
            new_center = new_box.mean(axis=0)[0]
            
            # Check distance to existing boxes (more lenient)
            is_duplicate = False
            for existing_center in existing_centers:
                dist = np.linalg.norm(new_center - existing_center)
                # Only filter if very close (same location)
                if dist < 20:  # Only filter true duplicates
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(new_box)
        
        return filtered
    
    def _detect_missed_watermarks_detections(self, existing_boxes: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect missed watermarks in complex backgrounds near existing detections.
        Uses conservative approach: only searches near known text regions.
        """
        dets: List[Dict[str, Any]] = []
        
        # Create search zones around existing detections
        search_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        buffer = 100  # Search within 100px of existing detections
        
        for box in existing_boxes:
            box_2d = box.reshape(-1, 2)
            x_min = max(0, int(box_2d[:, 0].min()) - buffer)
            y_min = max(0, int(box_2d[:, 1].min()) - buffer)
            x_max = min(self.width, int(box_2d[:, 0].max()) + buffer)
            y_max = min(self.height, int(box_2d[:, 1].max()) + buffer)
            search_mask[y_min:y_max, x_min:x_max] = 255
        
        # Identify complex background regions within search zones
        complex_mask = self._identify_complex_regions()
        search_mask = cv2.bitwise_and(search_mask, complex_mask)
        
        if np.count_nonzero(search_mask) == 0:
            return dets
        
        # Local enhancement only in search zones
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Aggressive CLAHE for complex regions
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Mask to search zone only
        masked_enhanced = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=search_mask)
        
        # Lower threshold for missed watermarks
        results = self.reader.readtext(
            masked_enhanced,
            low_text=0.15,
            text_threshold=0.35,
            link_threshold=0.25
        )
        
        # Validate each detection as watermark
        for detection in results:
            if not detection or len(detection) < 2:
                continue
            box = np.array(detection[0], dtype=np.int32).reshape((-1, 1, 2))
            text = detection[1] if len(detection) > 1 else ""
            confidence = detection[2] if len(detection) > 2 else 0.0
            
            # Verify watermark characteristics
            if self._is_likely_watermark(box, confidence):
                dets.append({"box": box, "text": text, "conf": float(confidence)})
        
        print(f"  Complex background (validated): {len(dets)} regions")
        return dets
    
    def _identify_complex_regions(self) -> np.ndarray:
        """Identify regions with complex multi-color backgrounds"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # High saturation variance = multiple colors
        sat = hsv[:, :, 1]
        sat_mean = cv2.blur(sat.astype(np.float32), (15, 15))
        sat_sq_mean = cv2.blur((sat.astype(np.float32) ** 2), (15, 15))
        sat_std = np.sqrt(np.maximum(sat_sq_mean - sat_mean ** 2, 0))
        
        # High value variance = brightness changes
        val = hsv[:, :, 2]
        val_mean = cv2.blur(val.astype(np.float32), (15, 15))
        val_sq_mean = cv2.blur((val.astype(np.float32) ** 2), (15, 15))
        val_std = np.sqrt(np.maximum(val_sq_mean - val_mean ** 2, 0))
        
        # Combine: complex = high color variance
        complex_mask = (
            (sat_std > 25) | (val_std > 35)
        ).astype(np.uint8) * 255
        
        # Dilate slightly to include boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        complex_mask = cv2.dilate(complex_mask, kernel, iterations=1)
        
        return complex_mask
    
    def _is_likely_watermark(self, box: np.ndarray, confidence: float) -> bool:
        """
        Verify if detected region is likely a watermark.
        Watermarks typically have: gray color, semi-transparent, edge position, low contrast.
        """
        box_2d = box.reshape(-1, 2)
        x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
        x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        roi = self.image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return False
        
        # Check 1: Gray color (watermarks are typically gray/neutral)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        a, b = lab[:, :, 1], lab[:, :, 2]
        a_std = np.std(a)
        b_std = np.std(b)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        # Gray = low saturation, near neutral (128 in LAB)
        is_gray = (a_std < 15 and b_std < 15 and 
                   abs(a_mean - 128) < 20 and abs(b_mean - 128) < 20)
        
        # Check 2: Low contrast (semi-transparent watermark)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_roi)
        is_low_contrast = contrast < 50  # Watermarks have lower contrast than real text
        
        # Check 3: Edge position (watermarks often near edges)
        edge_dist = min(x_min, y_min, self.width - x_max, self.height - y_max)
        is_near_edge = edge_dist < max(self.width, self.height) * 0.15
        
        # Check 4: OCR confidence (watermarks may have lower confidence)
        has_reasonable_confidence = confidence > 0.2
        
        # Must pass at least 2 checks to be considered watermark
        score = sum([is_gray, is_low_contrast, is_near_edge, has_reasonable_confidence])
        
        return score >= 2
    
    def generate_mask(
        self,
        expansion: int = 3,
        mode: str = "smart",
        shrink: int = 0,
        passes: int = 1,
        detect: str = "raw",
        rect_refine: str = "stroke",
        bridge: int = 1,
    ) -> np.ndarray:
        """
        Generate mask from detected text regions
        
        中文解释（mask 的作用与实现机制）：
        - mask 是最终输出的二值图（uint8，H×W），255 表示“要去除/修复的水印区域”。
        - 生成流程分两段：
          1) detection：先得到 boxes（OCR 检出的候选文字框）；
          2) masking：再把每个 box 转为更贴字的像素级 mask，并做少量形态学处理补齐笔画。
        - passes>1 时会启用“二次补漏”：每一轮用当前 mask 在“检测用图”上做一次 OpenCV inpaint，
          再跑下一轮检测，最终 mask 合并（注意：inpaint 只用于检测，不会修改原图输出）。
        """
        passes = max(1, int(passes))
        shrink = max(0, int(shrink))

        final_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # 暂存原图：passes>1 时会用 inpaint 生成“检测用图”，但不改变最终输出用的原图
        orig_image = self.image
        orig_h, orig_w = self.height, self.width

        try:
            working_image = orig_image
            for p in range(passes):
                self.image = working_image
                self.height, self.width = self.image.shape[:2]

                # 1) detection：得到候选 boxes
                if detect == "filtered":
                    dets = self.detect_text_detections(low_text=0.25, enable_template=False)
                    boxes = [d["box"] for d in dets]
                    self._dominant_watermark_angle = self._compute_dominant_angle(boxes)
                else:
                    boxes = self.detect_text_boxes_raw(low_text=0.25)

                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                if boxes:
                    print(f"🎨 Generating {mode} mask (pass {p+1}/{passes}, {len(boxes)} regions)...")
        
                rect_count = 0
                contour_count = 0

                if mode == "rect":
                    poly_union = np.zeros((self.height, self.width), dtype=np.uint8)
                    stroke_union = np.zeros((self.height, self.width), dtype=np.uint8)
                    rect_fallback = np.zeros((self.height, self.width), dtype=np.uint8)
                    box_wh: List[Tuple[int, int]] = []
        
                    for box in boxes:
                        cv2.fillPoly(poly_union, [box], 255)
                        rect_count += 1

                        if rect_refine == "stroke":
                            strict = self._is_complex_background(box)
                            stroke = self._extract_text_contour(box, strict=strict)
                            stroke_union = cv2.bitwise_or(stroke_union, stroke)

                            pts = box.reshape(-1, 2)
                            bw = int(pts[:, 0].max() - pts[:, 0].min())
                            bh = int(pts[:, 1].max() - pts[:, 1].min())
                            if bw > 0 and bh > 0:
                                box_wh.append((bw, bh))

                            # If stroke extraction fails for this box, fallback to polygon for this box only
                            if np.count_nonzero(stroke) < 30:
                                cv2.fillPoly(rect_fallback, [box], 255)

                    if rect_refine == "stroke":
                        # 🎯 桥接文字间隙 - 解决"文字间隙的残留"问题
                        # 在rect+stroke模式下，单个字符轮廓已提取，但字符间细小间隙仍未桥接
                        # 使用角度感知的形态学闭运算，沿着水印主方向精准连接

                        dom_angle = getattr(self, "_dominant_watermark_angle", None)
                        if bridge and box_wh:
                            # 📏 动态计算核尺寸：基于文字框的实际尺寸统计
                            ws = [wh[0] for wh in box_wh]  # 所有文字框的宽度列表
                            hs = [wh[1] for wh in box_wh]  # 所有文字框的高度列表
                            mw = float(np.median(ws))     # 文字框平均宽度 (median更稳健)
                            mh = float(np.median(hs))     # 文字框平均高度

                            # 🔧 自适应核参数计算:
                            # k_long: 沿着文字方向的长轴，覆盖字符间距 (mw的30%)
                            # k_short: 垂直文字方向的短轴，避免过度扩张 (mh的10%)
                            # 限制范围确保合理尺寸: k_long[9-45], k_short[3-11]
                            k_long = int(np.clip(mw * 0.30, 9, 45))
                            k_short = int(np.clip(mh * 0.10, 3, 11))

                            # 执行角度感知桥接闭运算
                            stroke_union = self._close_along_angle(stroke_union, dom_angle, k_long, k_short)

                        # Slight dilation to cover anti-aliasing around strokes
                        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        stroke_union = cv2.dilate(stroke_union, dk, iterations=1)

                        # Output is primarily stroke_union (tight like v12),
                        # plus a minimal fallback for boxes where stroke extraction failed.
                        # Shrink fallback polygons a bit to avoid huge blocks.
                        if np.count_nonzero(rect_fallback) > 0:
                            fk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                            rect_fallback = cv2.erode(rect_fallback, fk, iterations=1)
                        mask = cv2.bitwise_or(stroke_union, rect_fallback)
                    else:
                        mask = poly_union

                    print(f"  📦 Rect: {rect_count} (refine={rect_refine})")

                else:
                    for box in boxes:
                        strict = (mode == "precise") or self._is_complex_background(box)
                        contour_mask = self._extract_text_contour(box, strict=strict)
                        mask = cv2.bitwise_or(mask, contour_mask)
                        contour_count += 1
                    print(f"  🎯 Contour: {contour_count}")

                # Noise cleanup (always safe after contour extraction)
                mask = self._cleanup_mask_noise(mask)

                # Shrink (erode) to avoid hurting image elements
                if shrink > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink * 2 + 1, shrink * 2 + 1))
                    mask = cv2.erode(mask, k, iterations=1)

                # Merge into final mask in original coordinates (same size)
                final_mask = cv2.bitwise_or(final_mask, mask)

                # Prepare next pass: inpaint current working_image using newly found mask
                # NOTE: Inpainting is ONLY for detection in the next pass, not the final output.
                if p < passes - 1:
                    working_image = cv2.inpaint(working_image, mask, 3, cv2.INPAINT_TELEA)

        finally:
            self.image = orig_image
            self.height, self.width = orig_h, orig_w

        # 主体保护：只删除人物等主体的“强边缘”像素，尽量保留人物身上的斜向水印笔画
        dom_angle = getattr(self, "_dominant_watermark_angle", None)
        final_mask = self._remove_subject_components(final_mask, dom_angle)

        # expansion：最终轻微膨胀保证覆盖完整笔画
        if expansion > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        coverage = np.count_nonzero(final_mask) / (self.height * self.width) * 100
        print(f"💾 Mask coverage: {coverage:.2f}%")

        return final_mask
    
    def _extract_text_contour(self, box: np.ndarray, strict: bool = False) -> np.ndarray:
        """
        在OCR框内提取水印轮廓。
        核心：通过形状特征区分文字（细长、空洞、边缘密集）和物体（紧凑、实心、纯色）。
        
        关键原则：大段纯色区域一定不是文字，需要避开。

        中文补充（为什么能“贴字”且尽量不误伤）：
        - 我们不直接把整个 box 填满，而是希望只把“笔画像素”标出来；
        - 用局部背景差分 diff = |roi - blur(roi)|：
          - 半透明水印：diff 通常是“中等强度差异”（能看到笔画但不如主体边缘那么强）；
          - 主体轮廓/衣服边界：diff 往往非常强（高梯度）。
        - 因此采用 diff band：只保留 [thresh, high_cut] 区间的像素，抑制强边缘；
        - 再加上 Canny 边缘（同样用 high_cut 抑制强边缘），让笔画更连贯；
        - 对较大的连通域再用“方向约束”(PCA 主方向)过滤：与水印主角度差太多的更像主体结构而非水印。
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        box_2d = box.reshape(-1, 2)
        x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
        x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
        
        if x_max - x_min < 5 or y_max - y_min < 5:
            return mask
        
        roi_gray = cv2.cvtColor(self.image[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        h, w = roi_gray.shape
        roi_area = h * w
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # 1. 局部对比度：提取半透明水印笔画，同时抑制人物/物体强边缘
        blur_size = max(11, min(h, w) // 3) | 1
        local_bg = cv2.blur(roi_gray, (blur_size, blur_size))
        diff = cv2.absdiff(roi_gray, local_bg)
        
        diff_p80 = np.percentile(diff, 80)
        # lower base threshold for faint watermarks; still adaptive
        thresh = max(5, diff_p80 * 0.45)

        # Upper bound on diff to suppress strong object edges.
        # In complex backgrounds (strict), be more aggressive.
        high_cut = np.percentile(diff, 90 if strict else 96)
        high_cut = max(high_cut, thresh + 1)

        diff_band = ((diff >= thresh) & (diff <= high_cut)).astype(np.uint8) * 255
        contrast_mask = diff_band

        # Add edges for stroke continuity, but only if they are not too strong
        edges = cv2.Canny(roi_gray, 30, 120)
        edges = cv2.bitwise_and(edges, (diff <= high_cut).astype(np.uint8) * 255)
        contrast_mask = cv2.bitwise_or(contrast_mask, edges)
        contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 2. 连通区域分析
        contours, _ = cv2.findContours(contrast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_mask = np.zeros_like(contrast_mask)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3:
                continue
            
            area_ratio = area / roi_area
            
            # 计算形状特征
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(1, hull_area)  # 实心度
            complexity = perimeter / max(1, np.sqrt(area))  # 周长复杂度
            
            # 额外：避免把人物/物体大块区域纳入mask（对水印来说异常）
            if area_ratio > 0.25:
                continue

            # 方向约束：对较大的连通域，要求大致符合水印主方向（减轻人物轮廓误标）
            dom = getattr(self, "_dominant_watermark_angle", None)
            if dom is not None and area > 120:
                x, y, bw, bh = cv2.boundingRect(cnt)
                ar = max(bw, bh) / max(1, min(bw, bh))
                if ar >= 2.0:
                    pts = cnt.reshape(-1, 2).astype(np.float32)
                    # PCA major axis
                    mean = pts.mean(axis=0, keepdims=True)
                    cov = np.cov((pts - mean).T)
                    vals, vecs = np.linalg.eig(cov)
                    major = vecs[:, int(np.argmax(vals))]
                    ang = float(np.degrees(np.arctan2(major[1], major[0])))
                    while ang > 45:
                        ang -= 90
                    while ang < -45:
                        ang += 90
                    if abs(ang - float(dom)) > 28:
                        continue

            # 只对较大区域进行严格过滤
            if area_ratio > 0.08:
                # 大区域：必须有文字特征
                is_text = (solidity < 0.5) or (complexity > 6)
                if not is_text:
                    continue
            elif area_ratio > 0.03:
                # 中等区域：也要检查
                is_text = (solidity < 0.6) or (complexity > 5)
                if not is_text:
                    continue
            # 小区域（<3%）直接保留
            
            cv2.drawContours(roi_mask, [cnt], -1, 255, -1)
        
        # 3. 微膨胀
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
        
        mask[y_min:y_max, x_min:x_max] = roi_mask
        return mask
    
    def _cleanup_mask_noise(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove isolated tiny fragments (noise), keep text strokes.
        Very lenient - only remove obviously too small components.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        # Very lenient threshold - only remove tiny dots
        min_area = 15
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            # Only skip extremely tiny (noise dots)
            if w < 3 and h < 3:
                continue
            
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        
        return clean_mask
    
    def _is_complex_background(self, box: np.ndarray) -> bool:
        """
        Check if region has complex background (needs precise contour)
        Complex = high variance, multi-color, or near edges
        """
        box_2d = box.reshape(-1, 2)
        x_min, y_min = max(0, int(box_2d[:, 0].min())), max(0, int(box_2d[:, 1].min()))
        x_max, y_max = min(self.width, int(box_2d[:, 0].max())), min(self.height, int(box_2d[:, 1].max()))
        
        if x_max <= x_min or y_max <= y_min:
            return False
        
        roi = self.image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return False
        
        # 1. High color variance = complex background
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat_std = np.std(hsv[:, :, 1])
        val_std = np.std(hsv[:, :, 2])
        
        if sat_std > 30 or val_std > 40:
            return True
        
        # 2. Has multiple distinct colors
        mean_sat = np.mean(hsv[:, :, 1])
        if mean_sat > 25:  # Not pure gray
            return True
        
        # 3. High edge density = detailed content
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if edge_ratio > 0.1:
            return True
        
        return False
    
    def save_mask(self, mask: np.ndarray, output_path: Path):
        """Save mask"""
        cv2.imwrite(str(output_path), mask)
        print(f"💾 Mask saved: {output_path}")
    
    def save_preview(self, mask: np.ndarray, output_path: Path, 
                     boxes: List[np.ndarray] = None):
        """Save preview with detected boxes"""
        overlay = self.image.copy()
        
        # Red mask overlay
        red_mask = np.zeros_like(self.image)
        red_mask[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.6, red_mask, 0.4, 0)
        
        # Draw detected boxes if available
        if boxes is None:
            boxes = self.detect_text_regions()
        
        for i, box in enumerate(boxes):
            # Draw box
            cv2.polylines(overlay, [box], True, (0, 255, 0), 2)
            # Add number
            pt = box[0][0]  # First point
            cv2.putText(overlay, f"#{i+1}", (int(pt[0]), int(pt[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Stats with semi-transparent background
        coverage = np.count_nonzero(mask) / (self.height * self.width) * 100
        
        # Create semi-transparent rectangle for text background
        stats_overlay = overlay.copy()
        cv2.rectangle(stats_overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        overlay = cv2.addWeighted(stats_overlay, 0.4, overlay, 0.6, 0)
        
        cv2.putText(overlay, f"Coverage: {coverage:.2f}%", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Text regions: {len(boxes)}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), overlay)
        print(f"📷 Preview saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OCR-Based Watermark Detection (Most Accurate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Installation:
  pip install easyocr

Examples:
  # Detect and generate mask
  python detect_with_ocr.py -i sample.jpg
  
  # Expand mask for better coverage
  python detect_with_ocr.py -i sample.jpg --expand 10
  
Advantages:
  ✅ Detects diagonal/rotated text automatically
  ✅ Works with Chinese characters
  ✅ High accuracy for text watermarks
  ✅ No manual angle adjustment needed
  
Notes:
  - First run downloads models (~100MB)
  - Slower than OpenCV methods but more accurate
  - Best for text watermarks
        """
    )
    
    parser.add_argument('-i', '--input', type=Path, required=True,
                       help='Input image')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output mask path (default: input_mask.png)')
    parser.add_argument('--expand', type=int, default=3,
                       help='Expand mask by N pixels (default: 3)')
    parser.add_argument('--mode', choices=['smart', 'rect', 'precise'], default='smart',
                       help='Mask mode: smart=auto (default), rect=rectangle, precise=contour')
    parser.add_argument('--detect', choices=['raw', 'filtered'], default='raw',
                       help="Detection mode: raw=high recall (330+ regions), filtered=fewer false positives")
    parser.add_argument('--rect-refine', choices=['none', 'stroke'], default='stroke',
                       help="For --mode rect: refine polygons into stroke-like mask to fill character gaps (default: stroke)")
    parser.add_argument('--bridge', type=int, default=1,
                       help="Apply oriented closing to bridge small gaps between characters (default: 1)")
    parser.add_argument('--shrink', type=int, default=1,
                       help='Erode mask by N pixels to reduce over-marking (default: 1)')
    parser.add_argument('--passes', type=int, default=1,
                       help='Run N detection passes; uses OpenCV inpaint between passes for detection-only to catch residuals (default: 1)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Do not generate preview')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
        return
    
    if not EASYOCR_AVAILABLE:
        print("❌ EasyOCR not installed!")
        print("\nInstall with:")
        print("  pip install easyocr")
        print("\nOr use OpenCV-based detector:")
        print("  python detect_and_mask.py -i sample.jpg")
        return
    
    # Output paths
    if args.output:
        mask_path = args.output
    else:
        mask_path = args.input.parent / f"{args.input.stem}_mask.png"
    
    preview_path = mask_path.parent / f"{mask_path.stem}_preview.jpg"
    
    print("=" * 60)
    print("OCR-Based Watermark Detection")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {mask_path}")
    print()
    
    try:
        # Detect and generate mask
        detector = OCRWatermarkDetector(args.input)
        boxes = detector.detect_text_regions()
        mask = detector.generate_mask(
            expansion=args.expand,
            mode=args.mode,
            shrink=args.shrink,
            passes=args.passes,
            detect=args.detect,
            rect_refine=args.rect_refine,
            bridge=args.bridge,
        )
        
        # Save
        detector.save_mask(mask, mask_path)
        if not args.no_preview:
            detector.save_preview(mask, preview_path, boxes)
        
        print()
        print("=" * 60)
        print("Next: Remove watermarks")
        print("=" * 60)
        print(f"  export KMP_DUPLICATE_LIB_OK=TRUE")
        print(f"  iopaint run --model=lama --device=cpu \\")
        print(f"    --image={args.input} --mask={mask_path} --output=./output")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

