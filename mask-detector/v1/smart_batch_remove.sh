#!/bin/bash

# ==========================================
# 一键批量去除水印完整流程（智能版）
# 支持自动检测水印位置
# ==========================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 水印检测模式
# - "auto": 自动检测水印位置
# - "interactive": 交互式选择水印位置  
# - "manual": 手动指定坐标
DETECT_MODE="auto"

# 手动模式下的水印区域（相对坐标 0-1）
# 格式: X1 Y1 X2 Y2
WATERMARK_REGION="0.8 0.9 1.0 1.0"

# 样本图片（用于自动检测，留空则使用第一张图片）
SAMPLE_IMAGE=""

# IOPaint 模型选择
# 可选: lama, mat, fcf, sd1.5等
MODEL="fcf"

# 设备选择: cpu 或 cuda
DEVICE="cpu"

# ================================================

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}批量去除水印 - 智能版${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# 步骤0: 检查目录
echo -e "${YELLOW}[步骤 0/4]${NC} 检查输入目录..."
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

# 确定样本图片
if [ -z "$SAMPLE_IMAGE" ]; then
    SAMPLE_IMAGE=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -1)
    echo -e "${BLUE}📷 使用样本图片: $(basename "$SAMPLE_IMAGE")${NC}"
fi

echo ""

# 步骤1: 检测水印位置
if [ "$DETECT_MODE" = "auto" ]; then
    echo -e "${YELLOW}[步骤 1/4]${NC} 自动检测水印位置..."
    
    if [ ! -f "detect_watermark.py" ]; then
        echo -e "${RED}❌ 错误: detect_watermark.py 脚本不存在${NC}"
        exit 1
    fi
    
    # 自动检测并保存配置
    python3 detect_watermark.py \
        -i "$SAMPLE_IMAGE" \
        --visualize \
        --save-config watermark_config.json \
        --extract 0 \
        --template watermark_template.png
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 水印检测失败${NC}"
        echo -e "${YELLOW}💡 建议使用交互式模式: 将 DETECT_MODE 改为 'interactive'${NC}"
        exit 1
    fi
    
    # 从配置文件读取坐标
    if [ -f "watermark_config.json" ]; then
        # 提取第一个检测结果的坐标
        REGION_JSON=$(python3 -c "import json; config=json.load(open('watermark_config.json')); bbox=config['detections'][0]['relative_bbox']; print(' '.join(map(str, bbox)))" 2>/dev/null)
        if [ -n "$REGION_JSON" ]; then
            WATERMARK_REGION="$REGION_JSON"
            echo -e "${GREEN}✅ 检测到水印坐标: $WATERMARK_REGION${NC}"
        fi
    fi
    
elif [ "$DETECT_MODE" = "interactive" ]; then
    echo -e "${YELLOW}[步骤 1/4]${NC} 交互式选择水印位置..."
    
    if [ ! -f "detect_watermark.py" ]; then
        echo -e "${RED}❌ 错误: detect_watermark.py 脚本不存在${NC}"
        exit 1
    fi
    
    # 交互式选择
    python3 detect_watermark.py \
        -i "$SAMPLE_IMAGE" \
        --interactive \
        --save-config watermark_config.json \
        --extract 0 \
        --template watermark_template.png
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 用户取消或选择失败${NC}"
        exit 1
    fi
    
    # 从配置文件读取坐标
    if [ -f "watermark_config.json" ]; then
        REGION_JSON=$(python3 -c "import json; config=json.load(open('watermark_config.json')); bbox=config['detections'][0]['relative_bbox']; print(' '.join(map(str, bbox)))" 2>/dev/null)
        if [ -n "$REGION_JSON" ]; then
            WATERMARK_REGION="$REGION_JSON"
            echo -e "${GREEN}✅ 选择的水印坐标: $WATERMARK_REGION${NC}"
        fi
    fi
    
else
    echo -e "${YELLOW}[步骤 1/4]${NC} 使用手动指定的坐标..."
    echo -e "${BLUE}📍 水印坐标: $WATERMARK_REGION${NC}"
fi

echo ""

# 步骤2: 生成 Masks
echo -e "${YELLOW}[步骤 2/4]${NC} 批量生成 Masks..."
if [ ! -f "generate_masks.py" ]; then
    echo -e "${RED}❌ 错误: generate_masks.py 脚本不存在${NC}"
    exit 1
fi

# 使用模板匹配或坐标
if [ -f "watermark_template.png" ] && [ "$DETECT_MODE" != "manual" ]; then
    echo -e "${BLUE}🎯 使用模板匹配模式${NC}"
    python3 generate_masks.py \
        -i "$INPUT_DIR" \
        -o "$MASK_DIR" \
        --template watermark_template.png
else
    echo -e "${BLUE}🎯 使用坐标模式${NC}"
    python3 generate_masks.py \
        -i "$INPUT_DIR" \
        -o "$MASK_DIR" \
        --region $WATERMARK_REGION
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Mask 生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Masks 生成完成${NC}"
echo ""

# 步骤3: 批量处理图片
echo -e "${YELLOW}[步骤 3/4]${NC} 批量去除水印..."
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

# 步骤4: 清理临时文件（可选）
echo -e "${YELLOW}[步骤 4/4]${NC} 清理..."
# 可以选择是否删除中间文件
# rm -f watermark_config.json watermark_template.png
echo -e "${GREEN}✅ 保留了检测配置和模板文件，可用于下次处理${NC}"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✅ 完成！${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "📁 处理结果保存在: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "📁 生成的 Masks 保存在: ${GREEN}$MASK_DIR${NC}"
if [ "$DETECT_MODE" != "manual" ]; then
    echo -e "📁 检测配置保存在: ${GREEN}watermark_config.json${NC}"
    if [ -f "watermark_template.png" ]; then
        echo -e "📁 水印模板保存在: ${GREEN}watermark_template.png${NC}"
    fi
    echo -e "📁 检测可视化结果: ${GREEN}$(basename "$SAMPLE_IMAGE" | sed 's/\.[^.]*$/_detected.jpg/')${NC}"
fi
echo ""
echo -e "${YELLOW}💡 提示:${NC}"
echo "   - 检查 $OUTPUT_DIR 中的结果"
if [ "$DETECT_MODE" != "manual" ]; then
    echo "   - 查看 $(basename "$SAMPLE_IMAGE" | sed 's/\.[^.]*$/_detected.jpg/') 确认检测是否准确"
fi
echo "   - 如果效果不理想:"
echo "     • 使用交互式模式: 将 DETECT_MODE 改为 'interactive'"
echo "     • 调整 watermark_config.json 中的坐标"
echo "     • 尝试不同的模型: mat 或 fcf"
echo ""


