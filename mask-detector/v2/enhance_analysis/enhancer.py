"""图片增强模块 - 基于 IOPaint RealESRGAN 插件"""
import os
import sys
import cv2

# 添加 iopaint 到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class ImageEnhancer:
    """RealESRGAN 图片增强器（单例模式，懒加载）"""
    
    _instance = None
    
    def __new__(cls, device='cpu'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device='cpu'):
        if self._initialized:
            return
        self.device = device
        self._model = None
        self._initialized = True
    
    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """加载 RealESRGAN 模型"""
        from iopaint.plugins.realesrgan import RealESRGANUpscaler
        from iopaint.schema import RealESRGANModel
        
        print("🚀 加载 RealESRGAN 模型...")
        return RealESRGANUpscaler(
            name=RealESRGANModel.realesr_general_x4v3,
            device=self.device,
            no_half=(self.device == 'cpu')
        )
    
    def enhance(self, img_path: str, output_path: str = None, scale: float = 1) -> bool:
        """
        增强图片
        
        Args:
            img_path: 输入图片路径
            output_path: 输出路径（默认覆盖原图）
            scale: 缩放比例（1 = 仅增强不放大）
        
        Returns:
            是否成功
        """
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        enhanced = self.model.forward(img, scale=scale)
        cv2.imwrite(output_path or img_path, enhanced)
        return True

