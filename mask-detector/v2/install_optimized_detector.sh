#!/bin/bash
# 优化版水印检测器依赖安装脚本
# 使用 uv 包管理器进行安装

set -e  # 遇到错误立即退出

echo "🚀 开始安装优化版水印检测器依赖..."

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ 需要先安装 uv 包管理器"
    echo "安装方法: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📦 使用 uv 安装依赖..."
uv pip install -r requirements_optimized_detector.txt

echo "🔧 检查并修复版本兼容性..."

# 检查 NumPy 版本
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0")

# 比较版本号 (简化版)
if [[ "$NUMPY_VERSION" =~ ^2\. ]]; then
    echo "⚠️ 检测到 NumPy $NUMPY_VERSION，可能与某些包不兼容"
    echo "🔧 降级 NumPy 到兼容版本..."
    uv pip install "numpy<2.0"
    echo "✅ NumPy 已降级"
fi

echo "🧪 运行测试验证安装..."
if python test_optimized_detector.py; then
    echo ""
    echo "🎉 安装成功！"
    echo ""
    echo "📖 使用方法："
    echo "  python watermark_detector_optimized.py -i input.jpg -o mask.png"
    echo "  python watermark_detector_optimized.py -i input.jpg -o mask.png --preview"
    echo ""
    echo "📚 更多信息请查看: README_OPTIMIZED_DETECTOR.md"
else
    echo ""
    echo "⚠️ 安装可能有问题，请检查错误信息"
    echo "💡 常见问题："
    echo "  - 网络问题导致AI模型下载失败（正常，会自动使用传统方法）"
    echo "  - 内存不足（尝试使用较小的图像）"
    exit 1
fi
