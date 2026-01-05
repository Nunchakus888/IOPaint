"""
水印去除与图片增强模块

Usage:
    python pipeline.py

Modules:
    - config: 统一配置管理
    - enhancer: RealESRGAN 图片增强
    - watermark_remover: 水印去除 + MAT二次修复
    - pipeline: 流程编排

最佳实践 - 处理水印残留:
    1. LaMa 快速去除（默认）
    2. MAT 二次修复（ENABLE_REFINE=True，纹理生成能力更强）
    3. RealESRGAN 增强（ENABLE_ENHANCE=True，提升整体画质）
"""
from .config import *
from .enhancer import ImageEnhancer
from .watermark_remover import batch_remove_watermarks, remove_watermark, refine_with_mat

