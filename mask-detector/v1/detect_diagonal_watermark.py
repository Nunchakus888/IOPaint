#!/usr/bin/env python3
"""
Diagonal Repeated Watermark Detector
Optimized for repeated diagonal text watermarks like "邀请app 1000万"
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


class DiagonalWatermarkDetector:
    """Detect repeated diagonal watermarks using pattern analysis"""
    
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.image.shape[:2]
        
        # Enhanced for watermark detection
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        self.enhanced = clahe.apply(self.gray)
    
    def detect_dominant_angle(self) -> float:
        """Detect dominant diagonal angle of watermarks"""
        print("  Detecting watermark angle...")
        
        # Edge detection
        edges = cv2.Canny(self.enhanced, 30, 100)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                               minLineLength=40, maxLineGap=20)
        
        if lines is None:
            print("  No lines detected, using default angle -30°")
            return -30.0
        
        # Collect angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Normalize to -45 to +45
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            angles.append(angle)
        
        # Find dominant angle using histogram
        hist, bins = np.histogram(angles, bins=36, range=(-45, 45))
        dominant_idx = np.argmax(hist)
        dominant_angle = (bins[dominant_idx] + bins[dominant_idx + 1]) / 2
        
        print(f"  Dominant angle: {dominant_angle:.1f}°")
        return dominant_angle
    
    def generate_mask(self, angle: float = None, spacing: int = 150) -> np.ndarray:
        """
        Generate mask for diagonal repeated watermarks
        
        Args:
            angle: Watermark angle (auto-detect if None)
            spacing: Approximate spacing between watermark lines
        
        Returns:
            Binary mask
        """
        print("🔍 Generating diagonal watermark mask...")
        
        # Auto-detect angle if not provided
        if angle is None:
            angle = self.detect_dominant_angle()
        
        # Method 1: Rotate and detect horizontal patterns
        mask1 = self._detect_rotated_patterns(angle, spacing)
        
        # Method 2: Direct detection on enhanced image
        mask2 = self._detect_gray_text()
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Clean noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        coverage = np.count_nonzero(mask) / (self.height * self.width) * 100
        print(f"✅ Mask generated - Coverage: {coverage:.2f}%")
        
        return mask
    
    def _detect_rotated_patterns(self, angle: float, spacing: int) -> np.ndarray:
        """Detect patterns by rotating image to horizontal"""
        print(f"  Detecting rotated patterns at {angle:.1f}°...")
        
        # Rotate image to make diagonal watermarks horizontal
        center = (self.width // 2, self.height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        # Calculate new size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((self.height * sin) + (self.width * cos))
        new_h = int((self.height * cos) + (self.width * sin))
        
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Rotate enhanced image
        rotated = cv2.warpAffine(self.enhanced, rotation_matrix, (new_w, new_h),
                                borderValue=255)
        
        # Detect horizontal text patterns in rotated image
        mask_rotated = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # Use multiple thresholds
        for thresh in [140, 150, 160, 170, 180, 190]:
            # Detect text in specific gray range
            in_range = cv2.inRange(rotated, thresh - 30, thresh + 20)
            
            # Connect horizontally (text lines)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
            connected = cv2.morphologyEx(in_range, cv2.MORPH_CLOSE, kernel_h, iterations=1)
            
            mask_rotated = cv2.bitwise_or(mask_rotated, connected)
        
        # Rotate mask back
        inv_rotation_matrix = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        mask_back = cv2.warpAffine(mask_rotated, inv_rotation_matrix, (self.width, self.height),
                                   borderValue=0)
        
        return mask_back
    
    def _detect_gray_text(self) -> np.ndarray:
        """Detect gray text directly on enhanced image"""
        print("  Detecting gray text regions...")
        
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Detect specific gray ranges (common for watermarks)
        for lower, upper in [(120, 180), (140, 200), (160, 210)]:
            in_range = cv2.inRange(self.enhanced, lower, upper)
            
            # Keep only text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(in_range, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Filter by size
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 <= area <= 3000:  # Text-sized regions
                    cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask
    
    def save_mask(self, mask: np.ndarray, output_path: Path):
        """Save mask"""
        cv2.imwrite(str(output_path), mask)
        print(f"💾 Mask saved: {output_path}")
    
    def save_preview(self, mask: np.ndarray, output_path: Path):
        """Save preview"""
        overlay = self.image.copy()
        red_mask = np.zeros_like(self.image)
        red_mask[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.6, red_mask, 0.4, 0)
        
        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        
        # Stats
        coverage = np.count_nonzero(mask) / (self.height * self.width) * 100
        cv2.rectangle(overlay, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(overlay, f"Coverage: {coverage:.2f}%", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), overlay)
        print(f"📷 Preview saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagonal Repeated Watermark Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and generate mask
  python detect_diagonal_watermark.py -i sample.jpg
  
  # Specify watermark angle if known
  python detect_diagonal_watermark.py -i sample.jpg --angle -30
  
  # Adjust spacing for denser/sparser watermarks
  python detect_diagonal_watermark.py -i sample.jpg --spacing 100
        """
    )
    
    parser.add_argument('-i', '--input', type=Path, required=True,
                       help='Input image')
    parser.add_argument('-o', '--output', type=Path,
                       help='Output mask path (default: input_mask.png)')
    parser.add_argument('--angle', type=float,
                       help='Watermark angle in degrees (auto-detect if not specified)')
    parser.add_argument('--spacing', type=int, default=150,
                       help='Approximate spacing between watermark lines (default: 150)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Do not generate preview')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
        return
    
    # Output paths
    if args.output:
        mask_path = args.output
    else:
        mask_path = args.input.parent / f"{args.input.stem}_mask.png"
    
    preview_path = mask_path.parent / f"{mask_path.stem}_preview.jpg"
    
    print("=" * 60)
    print("Diagonal Repeated Watermark Detector")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {mask_path}")
    print()
    
    # Detect and generate mask
    detector = DiagonalWatermarkDetector(args.input)
    mask = detector.generate_mask(angle=args.angle, spacing=args.spacing)
    
    # Save
    detector.save_mask(mask, mask_path)
    if not args.no_preview:
        detector.save_preview(mask, preview_path)
    
    print()
    print("=" * 60)
    print("Next: Remove watermarks")
    print("=" * 60)
    print(f"  export KMP_DUPLICATE_LIB_OK=TRUE")
    print(f"  iopaint run --model=lama --device=cpu \\")
    print(f"    --image={args.input} --mask={mask_path} --output=./output")
    print("=" * 60)


if __name__ == '__main__':
    main()

