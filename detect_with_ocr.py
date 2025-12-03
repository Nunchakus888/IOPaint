#!/usr/bin/env python3
"""
OCR-Based Watermark Detection
Uses EasyOCR to detect text regions (including diagonal Chinese text)
Most accurate method for text watermarks
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

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
    
    def detect_text_regions(self, low_text: float = 0.3) -> List[np.ndarray]:
        """
        Detect all text regions including overlapping areas
        
        Args:
            low_text: Lower threshold for text detection
        """
        print("🔍 Detecting text regions...")
        
        all_boxes = []
        
        # 1. Main detection
        results = self.reader.readtext(
            self.image,
            low_text=low_text,
            text_threshold=0.5,
            link_threshold=0.3,
            width_ths=0.7
        )
        for detection in results:
            box = np.array(detection[0], dtype=np.int32).reshape((-1, 1, 2))
            all_boxes.append(box)
        print(f"  Main: {len(all_boxes)} regions")
        
        # 2. Enhanced detection for overlapping watermarks
        enhanced_boxes = self._detect_on_enhanced()
        all_boxes.extend(enhanced_boxes)
        
        # 3. Edge detection
        edge_boxes = self._detect_edge_text()
        all_boxes.extend(edge_boxes)
        
        # Remove duplicates
        all_boxes = self._remove_duplicate_boxes(all_boxes)
        
        print(f"✅ Total: {len(all_boxes)} text regions")
        return all_boxes
    
    def _detect_on_enhanced(self) -> List[np.ndarray]:
        """Detect watermarks on enhanced images (for different backgrounds)"""
        boxes = []
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: CLAHE enhanced (for low contrast areas)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(enhanced_bgr, low_text=0.25)
        for det in results:
            boxes.append(np.array(det[0], dtype=np.int32).reshape((-1, 1, 2)))
        
        # Method 2: Inverted (light watermarks on dark backgrounds)
        inverted = cv2.bitwise_not(self.image)
        results = self.reader.readtext(inverted, low_text=0.3)
        for det in results:
            boxes.append(np.array(det[0], dtype=np.int32).reshape((-1, 1, 2)))
        
        # Method 3: High contrast (black/white boundary areas)
        _, high_contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hc_bgr = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(hc_bgr, low_text=0.2)
        for det in results:
            boxes.append(np.array(det[0], dtype=np.int32).reshape((-1, 1, 2)))
        
        # Method 4: LAB L-channel (for blue-black boundary)
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        l_enhanced = clahe.apply(l_channel)
        l_bgr = cv2.cvtColor(l_enhanced, cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(l_bgr, low_text=0.25)
        for det in results:
            boxes.append(np.array(det[0], dtype=np.int32).reshape((-1, 1, 2)))
        
        # Method 5: Gamma correction (for white backgrounds)
        gamma = 0.5  # Darken to reveal light watermarks
        gamma_corrected = np.power(gray / 255.0, gamma) * 255
        gamma_bgr = cv2.cvtColor(gamma_corrected.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(gamma_bgr, low_text=0.25)
        for det in results:
            boxes.append(np.array(det[0], dtype=np.int32).reshape((-1, 1, 2)))
        
        print(f"  Enhanced: {len(boxes)} regions")
        return boxes
    
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
    
    def _detect_edge_text(self) -> List[np.ndarray]:
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
        
        boxes = []
        for detection in results:
            box = np.array(detection[0], dtype=np.int32)
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
                boxes.append(box.reshape((-1, 1, 2)))
        
        print(f"  Edge detection: {len(boxes)} regions")
        return boxes
    
    def generate_mask(self, expansion: int = 3, mode: str = 'precise') -> np.ndarray:
        """
        Generate mask from detected text regions
        
        Args:
            expansion: Expand mask by N pixels
            mode: 'rect'=rectangle, 'precise'=contour, 'strict'=minimal
        
        Returns:
            Binary mask
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Detect text boxes
        boxes = self.detect_text_regions()
        
        if not boxes:
            return mask
        
        print(f"🎨 Generating {mode} mask...")
        
        if mode == 'rect':
            # Simple rectangle fill
            for box in boxes:
                cv2.fillPoly(mask, [box], 255)
        else:
            # Extract precise text contours
            strict = (mode == 'strict')
            for box in boxes:
                contour_mask = self._extract_text_contour(box, strict=strict)
                mask = cv2.bitwise_or(mask, contour_mask)
        
        # Minimal expansion to ensure coverage
        if expansion > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion, expansion))
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        coverage = np.count_nonzero(mask) / (self.height * self.width) * 100
        print(f"💾 Mask coverage: {coverage:.2f}%")
        
        return mask
    
    def _extract_text_contour(self, box: np.ndarray, strict: bool = False) -> np.ndarray:
        """
        Extract precise watermark text contour (gray text only)
        
        Args:
            box: Detected text bounding box
            strict: If True, use stricter thresholds (less coverage but fewer false positives)
        
        Returns:
            Mask with only watermark text pixels marked
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Get bounding rect
        box_2d = box.reshape(-1, 2)
        x_min, y_min = box_2d.min(axis=0)
        x_max, y_max = box_2d.max(axis=0)
        
        # Clip to image bounds
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(self.width, int(x_max)), min(self.height, int(y_max))
        
        if x_max <= x_min or y_max <= y_min:
            return mask
        
        # Extract ROI
        roi = self.image[y_min:y_max, x_min:x_max]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Thresholds based on mode
        if strict:
            color_thresh = 15   # Stricter gray detection
            l_low, l_high = 100, 200
        else:
            color_thresh = 25   # More lenient
            l_low, l_high = 80, 220
        
        # Key insight: watermarks are GRAY (low color saturation)
        l, a, b = cv2.split(roi_lab)
        
        # Gray pixels: a and b near 128 (neutral)
        gray_mask = (
            (np.abs(a.astype(np.int16) - 128) < color_thresh) & 
            (np.abs(b.astype(np.int16) - 128) < color_thresh)
        ).astype(np.uint8) * 255
        
        # Brightness filter
        brightness_mask = (
            (l > l_low) & (l < l_high)
        ).astype(np.uint8) * 255
        
        # Combine: must be gray AND mid-brightness
        roi_mask = cv2.bitwise_and(gray_mask, brightness_mask)
        
        # Detect text-like edges within gray regions
        edges = cv2.Canny(roi_gray, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine with gray mask
        roi_mask = cv2.bitwise_and(roi_mask, edges)
        
        # Connect nearby pixels (text strokes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Place ROI mask back into full mask
        mask[y_min:y_max, x_min:x_max] = roi_mask
        
        return mask
    
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
        
        # Stats
        coverage = np.count_nonzero(mask) / (self.height * self.width) * 100
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
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
    parser.add_argument('--mode', choices=['rect', 'precise', 'strict'], default='precise',
                       help='Mask mode: rect=rectangle, precise=contour, strict=minimal (default: precise)')
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
        mask = detector.generate_mask(expansion=args.expand, mode=args.mode)
        
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

