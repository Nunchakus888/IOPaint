#!/bin/bash

# 工具集安装和环境检查脚本

echo "========================================"
echo "IOPaint 批量去水印工具集"
echo "环境检查"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查计数
PASS=0
FAIL=0

# 检查函数
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 已安装"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} $1 未安装"
        ((FAIL++))
        return 1
    fi
}

check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python 模块: $1"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} Python 模块: $1"
        ((FAIL++))
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} 文件: $1"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} 文件: $1"
        ((FAIL++))
        return 1
    fi
}

echo "=== 基础环境 ==="
check_command python3
check_command pip3

echo ""
echo "=== Python 模块 ==="
check_python_module cv2
check_python_module numpy
check_python_module PIL
check_python_module tqdm

echo ""
echo "=== IOPaint ==="
if python3 -c "import iopaint" 2>/dev/null; then
    VERSION=$(python3 -c "import iopaint; print(iopaint.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓${NC} IOPaint (版本: $VERSION)"
    ((PASS++))
else
    echo -e "${RED}✗${NC} IOPaint 未安装"
    echo -e "${YELLOW}  安装命令: pip3 install iopaint${NC}"
    ((FAIL++))
fi

echo ""
echo "=== 工具脚本 ==="
check_file "generate_masks.py"
check_file "visualize_watermark.py"
check_file "example_usage.py"
check_file "batch_remove_watermark.sh"
check_file "start_iopaint.sh"

echo ""
echo "=== 文档 ==="
check_file "00_START_HERE.md"
check_file "QUICK_START.md"
check_file "BATCH_TOOLS_README.md"
check_file "WATERMARK_REMOVAL_GUIDE.md"

echo ""
echo "========================================"
echo -e "检查结果: ${GREEN}$PASS 通过${NC} / ${RED}$FAIL 失败${NC}"
echo "========================================"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}🎉 所有检查通过！可以开始使用了。${NC}"
    echo ""
    echo "快速开始："
    echo "  1. 阅读快速指南: cat 00_START_HERE.md"
    echo "  2. 查看使用示例: cat QUICK_START.md"
    echo "  3. 运行一键脚本: ./batch_remove_watermark.sh"
else
    echo -e "${YELLOW}⚠️  有 $FAIL 项检查未通过，请先安装缺失的依赖。${NC}"
    echo ""
    echo "安装缺失的依赖："
    if ! command -v python3 &> /dev/null; then
        echo "  - Python 3: 访问 https://www.python.org/"
    fi
    if ! python3 -c "import cv2" 2>/dev/null; then
        echo "  - OpenCV: pip3 install opencv-python"
    fi
    if ! python3 -c "import numpy" 2>/dev/null; then
        echo "  - NumPy: pip3 install numpy"
    fi
    if ! python3 -c "import PIL" 2>/dev/null; then
        echo "  - Pillow: pip3 install Pillow"
    fi
    if ! python3 -c "import tqdm" 2>/dev/null; then
        echo "  - tqdm: pip3 install tqdm"
    fi
    if ! python3 -c "import iopaint" 2>/dev/null; then
        echo "  - IOPaint: pip3 install iopaint"
    fi
    echo ""
    echo "或一次性安装所有依赖："
    echo "  pip3 install opencv-python numpy Pillow tqdm iopaint"
fi

echo ""
echo "系统信息："
echo "  操作系统: $(uname -s)"
echo "  Python 版本: $(python3 --version 2>/dev/null || echo '未安装')"
echo "  当前目录: $(pwd)"

echo ""



