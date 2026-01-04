#!/usr/bin/env python3
"""
文字水印Mask生成器 - 参考 config.py 的旋转逻辑
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
from typing import List, Optional
import math
import os

# ==================== 配置区 ====================

def get_font(size: int) -> ImageFont:
    """获取字体"""
    candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for f in candidates:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


@dataclass
class TextSpan:
    """文字片段"""
    text: str
    size: int = 18


@dataclass
class Config:
    """水印配置"""
    # 第一行
    line1: List[TextSpan] = field(default_factory=lambda: [
        TextSpan("雪票、酒店、教练、摄影师、约玩", size=14),
    ])
    
    # 第二行（支持多样式拼接）
    line2: List[TextSpan] = field(default_factory=lambda: [
        TextSpan("滑呗", size=20),
        TextSpan(" app ", size=14),
        TextSpan("1000万", size=18),
        TextSpan("雪友的选择", size=14),
    ])
    
    # 布局参数（参考 config.py）
    angle: float = 25.0               # 旋转角度（正值=逆时针）
    horizontal_offset: int = 280      # 水平重复间隔
    line_spacing: int = 80            # 两行之间间距
    stagger: int = 120                # 错位偏移
    
    # 微调
    offset_x: int = 0
    offset_y: int = 0


# ==================== 核心实现 ====================

class TextMaskGenerator:
    def __init__(self, config: Config = None):
        self.cfg = config or Config()
    
    def generate(self, image: np.ndarray) -> np.ndarray:
        """生成mask - 参考 config.py 的旋转坐标算法"""
        h, w = image.shape[:2]
        cfg = self.cfg
        
        # 创建mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        # 预计算文字尺寸
        line1_w, line1_h = self._get_line_size(cfg.line1)
        line2_w, line2_h = self._get_line_size(cfg.line2)
        
        # 旋转参数
        angle = cfg.angle
        rad = math.radians(angle)
        cx, cy = w / 2, h / 2
        
        # 计算覆盖范围
        diag = math.sqrt(w**2 + h**2)
        step_x = cfg.horizontal_offset
        step_y = cfg.line_spacing * 2  # 两行一个周期
        steps = int(diag / min(step_x, step_y)) + 3
        
        print(f"   角度: {angle}°, 范围: {steps} steps")
        
        # 按 config.py 的方式铺满
        for i in range(-steps, steps):
            for j in range(-steps, steps):
                # 未旋转的网格位置
                x = i * step_x + j * cfg.stagger + cfg.offset_x
                y = j * step_y + cfg.offset_y
                
                # === 第一行 ===
                y1 = y
                # 围绕中心旋转坐标
                rx1 = cx + (x - cx) * math.cos(rad) - (y1 - cy) * math.sin(rad)
                ry1 = cy + (x - cx) * math.sin(rad) + (y1 - cy) * math.cos(rad)
                # 绘制
                self._draw_line(draw, cfg.line1, rx1 - line1_w/2, ry1 - line1_h/2)
                
                # === 第二行 ===
                y2 = y + cfg.line_spacing
                rx2 = cx + (x - cx) * math.cos(rad) - (y2 - cy) * math.sin(rad)
                ry2 = cy + (x - cx) * math.sin(rad) + (y2 - cy) * math.cos(rad)
                self._draw_line(draw, cfg.line2, rx2 - line2_w/2, ry2 - line2_h/2)
        
        return np.array(mask)
    
    def _get_line_size(self, spans: List[TextSpan]):
        """计算一行文字的尺寸"""
        total_w, max_h = 0, 0
        for span in spans:
            font = get_font(span.size)
            bbox = font.getbbox(span.text)
            total_w += bbox[2] - bbox[0]
            max_h = max(max_h, bbox[3] - bbox[1])
        return total_w, max_h
    
    def _draw_line(self, draw: ImageDraw, spans: List[TextSpan], x: float, y: float):
        """绘制一行多样式文字"""
        cur_x = x
        for span in spans:
            font = get_font(span.size)
            draw.text((cur_x, y), span.text, fill=255, font=font)
            bbox = font.getbbox(span.text)
            cur_x += bbox[2] - bbox[0]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='文字水印Mask生成器')
    parser.add_argument('image', help='输入图片')
    parser.add_argument('-o', '--output', help='输出mask路径')
    parser.add_argument('--preview', action='store_true', help='生成预览')
    args = parser.parse_args()
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法读取: {args.image}")
        return
    
    print(f"🎯 处理: {args.image}")
    
    gen = TextMaskGenerator()
    mask = gen.generate(image)
    
    output = args.output or args.image.replace('.', '_mask.')
    cv2.imwrite(output, mask)
    print(f"💾 Mask: {output}")
    
    if args.preview:
        preview = image.copy()
        preview[mask > 127] = [0, 255, 0]
        preview_path = output.replace('_mask', '_preview')
        cv2.imwrite(preview_path, preview)
        print(f"🖼️ Preview: {preview_path}")


if __name__ == '__main__':
    main()
