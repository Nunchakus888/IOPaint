#!/bin/bash

# ==========================================
# 一键批量去除水印完整流程
# ==========================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 解决 macOS OpenMP 冲突
export KMP_DUPLICATE_LIB_OK=TRUE

# ==================== 配置区 ====================

# 输入图片目录
INPUT_DIR="./images"

# Mask 输出目录
MASK_DIR="./masks"

# 处理结果输出目录
OUTPUT_DIR="./output"

# 水印区域（相对坐标 0-1）
# 格式: X1 Y1 X2 Y2
# 示例: 0.8 0.9 1.0 1.0 表示右下角 20%x10% 的区域
WATERMARK_REGION="0.8 0.9 1.0 1.0"

# IOPaint 模型选择
# 可选: lama, mat, fcf, sd1.5等
MODEL="lama"

# 设备选择: cpu 或 cuda
DEVICE="cpu"

# ================================================

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}批量去除水印 - 完整流程${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# 步骤1: 检查目录
echo -e "${YELLOW}[步骤 1/3]${NC} 检查输入目录..."
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ 错误: 输入目录不存在: $INPUT_DIR${NC}"
    echo -e "${YELLOW}💡 请创建目录并放入要处理的图片${NC}"
    exit 1
fi

# 统计图片数量
IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ 错误: 在 $INPUT_DIR 中没有找到图片文件${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 找到 $IMAGE_COUNT 张图片${NC}"
echo ""

# 步骤2: 生成 Masks
echo -e "${YELLOW}[步骤 2/3]${NC} 生成 Masks..."
if [ ! -f "generate_masks.py" ]; then
    echo -e "${RED}❌ 错误: generate_masks.py 脚本不存在${NC}"
    exit 1
fi

python3 generate_masks.py \
    -i "$INPUT_DIR" \
    -o "$MASK_DIR" \
    --region $WATERMARK_REGION

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Mask 生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Masks 生成完成${NC}"
echo ""

# 步骤3: 批量处理图片
echo -e "${YELLOW}[步骤 3/3]${NC} 批量去除水印..."
iopaint run \
    --model=$MODEL \
    --device=$DEVICE \
    --image="$INPUT_DIR" \
    --mask="$MASK_DIR" \
    --output="$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 批量处理失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✅ 完成！${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "📁 处理结果保存在: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "📁 生成的 Masks 保存在: ${GREEN}$MASK_DIR${NC}"
echo ""
echo -e "${YELLOW}💡 提示:${NC}"
echo "   - 如果效果不理想，可以调整 WATERMARK_REGION 参数"
echo "   - 使用 python3 generate_masks.py -i $INPUT_DIR --preview 查看图片尺寸"
echo "   - 可以尝试不同的模型，如 mat 或 fcf"
echo ""


