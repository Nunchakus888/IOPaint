#!/usr/bin/env python3
"""
批量生成固定位置水印的 Mask 工具
用于 IOPaint 批量去除水印
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class MaskGenerator:
    """Mask 生成器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化 Mask 生成器
        
        Args:
            output_dir: mask 输出目录
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_fixed_region_mask(
        self, 
        image_path: Path, 
        regions: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        在固定区域生成 mask
        
        Args:
            image_path: 图片路径
            regions: 区域列表，每个区域为 (x1, y1, x2, y2)
                    x1, y1: 左上角坐标
                    x2, y2: 右下角坐标
        
        Returns:
            mask 数组
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 在指定区域填充白色
        for x1, y1, x2, y2 in regions:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def generate_relative_region_mask(
        self,
        image_path: Path,
        regions: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        """
        使用相对位置生成 mask（推荐）
        
        Args:
            image_path: 图片路径
            regions: 相对区域列表，每个区域为 (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
                    值范围: 0.0 - 1.0
                    例如：(0.8, 0.9, 1.0, 1.0) 表示右下角 20%x10% 的区域
        
        Returns:
            mask 数组
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 根据相对位置计算实际坐标
        for x1_ratio, y1_ratio, x2_ratio, y2_ratio in regions:
            x1 = int(width * x1_ratio)
            y1 = int(height * y1_ratio)
            x2 = int(width * x2_ratio)
            y2 = int(height * y2_ratio)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def generate_template_matching_mask(
        self,
        image_path: Path,
        template_path: Path,
        threshold: float = 0.8
    ) -> np.ndarray:
        """
        使用模板匹配生成 mask（适用于水印图案固定的情况）
        
        Args:
            image_path: 图片路径
            template_path: 水印模板图片路径
            threshold: 匹配阈值 (0-1)，越高越严格
        
        Returns:
            mask 数组
        """
        img = cv2.imread(str(image_path))
        template = cv2.imread(str(template_path))
        
        if img is None or template is None:
            raise ValueError("无法读取图片或模板")
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        height, width = img.shape[:2]
        t_height, t_width = template_gray.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 模板匹配
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        
        # 在匹配位置绘制 mask
        for pt in zip(*locations[::-1]):
            cv2.rectangle(
                mask, 
                pt, 
                (pt[0] + t_width, pt[1] + t_height), 
                255, 
                -1
            )
        
        return mask
    
    def save_mask(self, mask: np.ndarray, output_path: Path):
        """保存 mask 到文件"""
        cv2.imwrite(str(output_path), mask)


def get_image_files(input_dir: Path) -> List[Path]:
    """获取目录中的所有图片文件"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in extensions and f.is_file()
    ]


def batch_generate_masks(
    input_dir: Path,
    output_dir: Path,
    regions: Optional[List[Tuple[float, float, float, float]]] = None,
    template_path: Optional[Path] = None,
    use_relative: bool = True
):
    """
    批量生成 masks
    
    Args:
        input_dir: 输入图片目录
        output_dir: mask 输出目录
        regions: 水印区域列表（相对或绝对坐标）
        template_path: 水印模板路径（可选）
        use_relative: 是否使用相对坐标
    """
    generator = MaskGenerator(output_dir)
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"错误：在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"Mask 将保存到: {output_dir}")
    
    for image_path in tqdm(image_files, desc="生成 masks"):
        try:
            # 根据不同模式生成 mask
            if template_path:
                mask = generator.generate_template_matching_mask(
                    image_path, 
                    template_path
                )
            elif use_relative and regions:
                mask = generator.generate_relative_region_mask(
                    image_path, 
                    regions
                )
            elif regions:
                # 将相对坐标转为绝对坐标（如果需要）
                img = cv2.imread(str(image_path))
                height, width = img.shape[:2]
                absolute_regions = [
                    (
                        int(width * r[0]), 
                        int(height * r[1]), 
                        int(width * r[2]), 
                        int(height * r[3])
                    )
                    for r in regions
                ]
                mask = generator.generate_fixed_region_mask(
                    image_path, 
                    absolute_regions
                )
            else:
                print("错误：必须指定 regions 或 template_path")
                return
            
            # 保存 mask，保持与原图相同的文件名
            output_path = output_dir / f"{image_path.stem}.png"
            generator.save_mask(mask, output_path)
            
        except Exception as e:
            print(f"\n处理 {image_path.name} 时出错: {e}")
    
    print(f"\n✅ 完成！共生成 {len(image_files)} 个 mask 文件")


def main():
    parser = argparse.ArgumentParser(
        description="批量生成固定位置水印的 Mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用相对位置（推荐）- 水印在右下角
  python generate_masks.py -i ./images -o ./masks --region 0.8 0.9 1.0 1.0
  
  # 多个水印区域 - 右下角和左上角
  python generate_masks.py -i ./images -o ./masks \\
    --region 0.8 0.9 1.0 1.0 \\
    --region 0.0 0.0 0.2 0.1
  
  # 使用模板匹配
  python generate_masks.py -i ./images -o ./masks --template watermark.png
  
  # 查看图片尺寸以便确定水印位置
  python generate_masks.py -i ./images --preview
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='输入图片目录'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='mask 输出目录（默认：输入目录/masks）'
    )
    
    parser.add_argument(
        '--region',
        nargs=4,
        type=float,
        action='append',
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help='水印区域（相对坐标 0-1），可多次指定。例如：--region 0.8 0.9 1.0 1.0'
    )
    
    parser.add_argument(
        '--template',
        type=Path,
        help='水印模板图片路径（用于模板匹配）'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='预览第一张图片的尺寸信息'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not args.input.exists():
        print(f"错误：输入目录不存在: {args.input}")
        return
    
    # 预览模式
    if args.preview:
        image_files = get_image_files(args.input)
        if image_files:
            img = cv2.imread(str(image_files[0]))
            height, width = img.shape[:2]
            print(f"\n📷 第一张图片: {image_files[0].name}")
            print(f"   尺寸: {width} x {height}")
            print(f"\n💡 常见水印位置示例:")
            print(f"   右下角 (20%x10%): --region 0.8 0.9 1.0 1.0")
            print(f"   左上角 (20%x10%): --region 0.0 0.0 0.2 0.1")
            print(f"   右上角 (20%x10%): --region 0.8 0.0 1.0 0.1")
            print(f"   左下角 (20%x10%): --region 0.0 0.9 0.2 1.0")
            print(f"   底部居中 (30%x8%): --region 0.35 0.92 0.65 1.0")
        return
    
    # 设置输出目录
    output_dir = args.output if args.output else args.input / 'masks'
    
    # 检查参数
    if not args.region and not args.template:
        print("错误：必须指定 --region 或 --template")
        print("使用 --preview 查看图片尺寸以确定水印位置")
        return
    
    # 转换 region 格式
    regions = [tuple(r) for r in args.region] if args.region else None
    
    # 批量生成
    batch_generate_masks(
        input_dir=args.input,
        output_dir=output_dir,
        regions=regions,
        template_path=args.template,
        use_relative=True
    )


if __name__ == '__main__':
    main()


