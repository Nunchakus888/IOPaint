# 🎯 智能水印检测使用指南

全新的智能水印检测工具，**无需手动指定坐标**！

---

## 🚀 核心功能

✨ **自动检测水印位置** - 使用计算机视觉算法自动找出水印
✨ **交互式选择** - 鼠标框选一次，应用所有图片
✨ **模板提取** - 自动提取水印作为模板，支持模板匹配
✨ **一键处理** - 从检测到批量处理全自动化

---

## 📖 三种使用方式

### 方式1: 自动检测（最智能）⭐⭐⭐

```bash
# 1. 运行智能检测工具
python3 detect_watermark.py -i images/sample.jpg --visualize

# 工具会自动检测并显示所有可能的水印位置
# 检查生成的 sample_detected.jpg 查看检测结果

# 2. 如果检测准确，提取为模板
python3 detect_watermark.py -i images/sample.jpg \
  --extract 0 \
  --template watermark_template.png \
  --save-config watermark_config.json

# 3. 批量生成 masks
python3 generate_masks.py -i ./images -o ./masks \
  --template watermark_template.png

# 或使用检测到的坐标
python3 generate_masks.py -i ./images -o ./masks \
  --region 0.85 0.92 1.0 1.0  # 从检测结果中复制

# 4. 批量处理
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint run --model=lama --device=cpu \
  --image=./images --mask=./masks --output=./output
```

**优点：** 完全自动，无需人工干预  
**适用：** 水印在角落或边缘，有明显特征

---

### 方式2: 交互式选择（最精确）⭐⭐⭐

```bash
# 1. 启动交互式工具
python3 detect_watermark.py -i images/sample.jpg --interactive

# 2. 在弹出的窗口中:
#    - 鼠标拖动框选水印区域
#    - 按 'c' 确认
#    - 按 'r' 重新选择
#    - 按 'q' 退出

# 3. 自动提取模板和保存坐标
python3 detect_watermark.py -i images/sample.jpg \
  --interactive \
  --extract 0 \
  --template watermark_template.png \
  --save-config watermark_config.json

# 4. 批量处理（同方式1步骤3-4）
```

**优点：** 精确度最高，一次框选应用所有  
**适用：** 所有场景，特别是水印位置不规则的情况

---

### 方式3: 一键智能处理（最简单）⭐⭐⭐

```bash
# 1. 编辑配置
nano smart_batch_remove.sh

# 修改这几行：
INPUT_DIR="./images"
DETECT_MODE="auto"        # 或 "interactive" 或 "manual"
SAMPLE_IMAGE=""           # 留空使用第一张图片

# 2. 运行
./smart_batch_remove.sh

# 完成！自动完成检测→生成masks→批量处理
```

**优点：** 一键搞定所有步骤  
**适用：** 新手，想要最简单的方案

---

## 🛠️ detect_watermark.py 详细用法

### 基础命令

```bash
# 自动检测（不保存）
python3 detect_watermark.py -i sample.jpg

# 自动检测并可视化
python3 detect_watermark.py -i sample.jpg --visualize

# 交互式选择
python3 detect_watermark.py -i sample.jpg --interactive

# 提取第1个检测结果为模板
python3 detect_watermark.py -i sample.jpg \
  --extract 0 \
  --template watermark.png

# 保存配置文件（包含所有检测结果）
python3 detect_watermark.py -i sample.jpg \
  --save-config config.json

# 完整流程（自动检测+可视化+提取+保存）
python3 detect_watermark.py -i sample.jpg \
  --visualize \
  --extract 0 \
  --template watermark.png \
  --save-config config.json
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-i, --input` | 输入样本图片 | `-i sample.jpg` |
| `--visualize` | 可视化检测结果 | 生成 `sample_detected.jpg` |
| `--interactive` | 交互式选择模式 | 鼠标框选 |
| `--extract N` | 提取第N个检测结果 | `--extract 0` (第1个) |
| `--template` | 模板输出路径 | `--template wm.png` |
| `--save-config` | 保存配置文件 | `--save-config config.json` |

---

## 🔍 检测原理

工具使用三种算法自动检测水印：

### 1. 角落检测
- 分析图片四个角落的区域
- 使用边缘检测识别有内容的区域
- **适用：** 角落水印（最常见）

### 2. 文字检测  
- 使用形态学操作检测文字区域
- 过滤太小或太大的区域
- **适用：** 文字类水印

### 3. 高亮检测
- 检测高亮度区域
- 识别白色或半透明水印
- **适用：** 亮色水印

### 自动合并
- 将重叠的检测结果合并
- 去除重复区域
- 按可信度排序

---

## 📊 检测结果示例

运行检测后，会输出：

```
🔍 开始自动检测水印...
  → 检测角落区域...
     找到 2 个角落水印
  → 检测文字区域...
     找到 1 个文字区域
  → 检测高亮区域...
     找到 0 个高亮区域

✅ 总共检测到 2 个可能的水印区域

============================================================
检测结果:
============================================================

区域 #1 (方法: corner)
  相对坐标: 0.8500 0.9200 1.0000 1.0000
  命令参数: --region 0.8500 0.9200 1.0000 1.0000

区域 #2 (方法: text)
  相对坐标: 0.0000 0.0000 0.1500 0.0800
  命令参数: --region 0.0000 0.0000 0.1500 0.0800
```

---

## 🎨 可视化结果

使用 `--visualize` 后，会生成 `xxx_detected.jpg`：

- 🔴 红色框：角落检测
- 🟢 绿色框：文字检测  
- 🔵 蓝色框：高亮检测
- 每个框有编号和坐标

![检测示例](sample_detected.jpg)

---

## 💡 最佳实践

### 1. 选择合适的样本图片

```bash
# 选择水印清晰、完整的图片作为样本
# 避免:
# - 水印被遮挡
# - 水印与背景颜色接近
# - 图片分辨率太低
```

### 2. 验证检测结果

```bash
# 1. 先可视化
python3 detect_watermark.py -i sample.jpg --visualize

# 2. 打开 sample_detected.jpg 检查
open sample_detected.jpg

# 3. 确认检测准确后再提取
```

### 3. 使用模板匹配还是坐标？

**模板匹配：**
- ✅ 适用于水印图案固定
- ✅ 可以处理位置略有偏移的情况
- ❌ 对图案变化敏感

**坐标模式：**
- ✅ 适用于位置绝对固定
- ✅ 处理速度快
- ❌ 位置偏移会失效

**建议：**
- 如果水印图案完全一致 → 使用模板
- 如果只是位置固定 → 使用坐标
- 不确定 → 两种都试试

---

## 🔧 调试和优化

### 检测不到水印？

```bash
# 1. 使用交互式模式手动选择
python3 detect_watermark.py -i sample.jpg --interactive

# 2. 检查图片
# - 水印是否在边缘？
# - 水印是否足够清晰？
# - 图片分辨率是否足够？

# 3. 尝试不同的样本图片
```

### 检测到太多区域？

```bash
# 查看可视化结果，选择正确的索引
python3 detect_watermark.py -i sample.jpg --visualize

# 提取特定的检测结果
python3 detect_watermark.py -i sample.jpg \
  --extract 0  # 使用第1个检测结果
```

### 检测位置不准确？

```bash
# 方法1: 使用交互式精确选择
python3 detect_watermark.py -i sample.jpg --interactive

# 方法2: 编辑配置文件微调坐标
nano watermark_config.json

# 方法3: 手动指定坐标
python3 visualize_watermark.py -i sample.jpg \
  --region 0.85 0.92 1.0 1.0
```

---

## 📋 完整工作流程

```bash
# === 第一步：检测水印 ===
python3 detect_watermark.py -i images/sample.jpg \
  --visualize \
  --extract 0 \
  --template watermark_template.png \
  --save-config watermark_config.json

# 检查检测结果
open images/sample_detected.jpg

# === 第二步：批量生成 masks ===
# 方法A: 使用模板
python3 generate_masks.py -i ./images -o ./masks \
  --template watermark_template.png

# 方法B: 使用坐标（从检测结果复制）
python3 generate_masks.py -i ./images -o ./masks \
  --region 0.8500 0.9200 1.0000 1.0000

# === 第三步：批量处理 ===
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint run --model=lama --device=cpu \
  --image=./images --mask=./masks --output=./output

# === 检查结果 ===
open output/
```

---

## 🎯 使用场景示例

### 场景1: 电商产品图

```bash
# 右下角logo水印
python3 detect_watermark.py -i product_sample.jpg --interactive
# 框选右下角logo → 按 c 确认

python3 generate_masks.py -i ./products -o ./masks \
  --region 0.82 0.88 0.98 0.98

iopaint run --model=lama --device=cuda \
  --image=./products --mask=./masks --output=./clean
```

### 场景2: 视频截图

```bash
# 多个水印：右下角时间 + 左上角台标
python3 detect_watermark.py -i frame_sample.jpg --visualize

# 查看检测结果，使用多个区域
python3 generate_masks.py -i ./frames -o ./masks \
  --region 0.85 0.92 1.0 1.0 \
  --region 0.0 0.0 0.12 0.08

iopaint run --model=mat --device=cuda \
  --image=./frames --mask=./masks --output=./clean
```

### 场景3: 固定图案水印

```bash
# 提取水印图案作为模板
python3 detect_watermark.py -i sample.jpg \
  --extract 0 \
  --template logo_watermark.png

# 使用模板匹配（适用于位置略有偏移）
python3 generate_masks.py -i ./images -o ./masks \
  --template logo_watermark.png

iopaint run --model=lama --device=cpu \
  --image=./images --mask=./masks --output=./clean
```

---

## ⚡ 快速命令参考

```bash
# 自动检测完整流程
python3 detect_watermark.py -i sample.jpg --visualize --extract 0 --template wm.png --save-config config.json

# 交互式选择完整流程
python3 detect_watermark.py -i sample.jpg --interactive --extract 0 --template wm.png

# 一键智能处理
./smart_batch_remove.sh  # 编辑配置后运行

# 使用模板批量处理
python3 generate_masks.py -i ./images -o ./masks --template wm.png
iopaint run --model=lama --device=cpu --image=./images --mask=./masks --output=./output

# 使用坐标批量处理
python3 generate_masks.py -i ./images -o ./masks --region 0.85 0.92 1.0 1.0
iopaint run --model=lama --device=cpu --image=./images --mask=./masks --output=./output
```

---

## 🆚 三种方式对比

| 特性 | 自动检测 | 交互式选择 | 手动坐标 |
|------|---------|-----------|---------|
| 便捷性 | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| 精确度 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 学习成本 | 低 | 低 | 中 |
| 适用场景 | 典型水印 | 所有场景 | 明确位置 |
| 推荐度 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 📝 配置文件格式

```json
{
  "image": "sample.jpg",
  "detections": [
    {
      "relative_bbox": [0.85, 0.92, 1.0, 1.0],
      "method": "corner"
    },
    {
      "relative_bbox": [0.0, 0.0, 0.15, 0.08],
      "method": "text"
    }
  ]
}
```

可以手动编辑此文件调整坐标。

---

## 🎓 总结

1. **新手推荐：** 使用 `smart_batch_remove.sh` 一键处理
2. **精确控制：** 使用 `detect_watermark.py --interactive` 交互式选择
3. **批量自动：** 使用 `detect_watermark.py --visualize` 自动检测后批处理

**现在你可以真正做到无需手动参数化，自动检测并批量去除水印了！** 🎉

---

更多信息请参考：
- `00_START_HERE.md` - 总体介绍
- `QUICK_START.md` - 快速上手
- `WATERMARK_REMOVAL_GUIDE.md` - 完整手册


