# IOPaint 批量去水印工具集

针对**固定位置水印**的完整批处理解决方案。

---

## 📦 工具清单

### 核心工具

| 文件 | 类型 | 功能 | 推荐场景 |
|------|------|------|----------|
| `generate_masks.py` | Python | 批量生成 mask | 命令行批处理 ⭐ |
| `example_usage.py` | Python | 简化配置脚本 | 快速上手 |
| `batch_remove_watermark.sh` | Shell | 一键完整流程 | 最简单方式 ⭐ |
| `visualize_watermark.py` | Python | 可视化水印位置 | 确定坐标 ⭐ |
| `start_iopaint.sh` | Shell | 启动 Web UI | 手动处理 |

### 文档

| 文件 | 内容 |
|------|------|
| `QUICK_START.md` | 快速开始指南（3分钟上手）⭐ |
| `WATERMARK_REMOVAL_GUIDE.md` | 完整使用手册 |
| `BATCH_TOOLS_README.md` | 本文件（工具总览）|

---

## 🚀 快速开始

### 方式1: 一键处理（最简单）

```bash
# 1. 准备图片
mkdir -p images
# 将图片放入 images/ 目录

# 2. 编辑配置
nano batch_remove_watermark.sh
# 修改 WATERMARK_REGION 参数

# 3. 运行
./batch_remove_watermark.sh
```

### 方式2: 分步处理（更灵活）

```bash
# 1. 查看图片尺寸
python3 generate_masks.py -i ./images --preview

# 2. 可视化水印位置
python3 visualize_watermark.py -i ./images/sample.jpg \
  --region 0.8 0.9 1.0 1.0

# 3. 批量生成 masks
python3 generate_masks.py -i ./images -o ./masks \
  --region 0.8 0.9 1.0 1.0

# 4. 批量处理
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint run --model=lama --device=cpu \
  --image=./images --mask=./masks --output=./output
```

### 方式3: Web UI（手动选择）

```bash
./start_iopaint.sh
# 访问 http://localhost:8080
```

---

## 📚 详细说明

### 1. `generate_masks.py` - Mask 生成器

**功能：** 批量生成固定位置水印的 mask

**核心特性：**
- ✅ 支持相对坐标（自动适配不同尺寸）
- ✅ 支持多个水印区域
- ✅ 支持模板匹配
- ✅ 进度条显示
- ✅ 错误处理

**基本用法：**

```bash
# 查看帮助
python3 generate_masks.py --help

# 查看图片尺寸
python3 generate_masks.py -i ./images --preview

# 生成 mask（单区域）
python3 generate_masks.py -i ./images -o ./masks \
  --region 0.8 0.9 1.0 1.0

# 生成 mask（多区域）
python3 generate_masks.py -i ./images -o ./masks \
  --region 0.8 0.9 1.0 1.0 \
  --region 0.0 0.0 0.2 0.1

# 使用模板匹配
python3 generate_masks.py -i ./images -o ./masks \
  --template watermark.png
```

**高级用法：**

```python
# 在 Python 代码中使用
from generate_masks import MaskGenerator
from pathlib import Path

generator = MaskGenerator(Path("./masks"))
mask = generator.generate_relative_region_mask(
    Path("image.jpg"),
    [(0.8, 0.9, 1.0, 1.0)]
)
generator.save_mask(mask, Path("./masks/mask.png"))
```

### 2. `visualize_watermark.py` - 可视化工具

**功能：** 在图片上直观显示水印区域

**使用场景：**
- 🎯 确定水印精确位置
- 🎯 验证坐标是否正确
- 🎯 调整区域大小

**用法：**

```bash
# 查看常用位置
python3 visualize_watermark.py --list

# 可视化单个区域
python3 visualize_watermark.py -i photo.jpg \
  --region 0.8 0.9 1.0 1.0

# 可视化多个区域
python3 visualize_watermark.py -i photo.jpg \
  --region 0.8 0.9 1.0 1.0 \
  --region 0.0 0.0 0.2 0.1

# 保存可视化结果
python3 visualize_watermark.py -i photo.jpg \
  --region 0.8 0.9 1.0 1.0 \
  -o preview.jpg
```

### 3. `example_usage.py` - 简化脚本

**功能：** 提供易于修改的配置式脚本

**适合人群：** Python 初学者，不熟悉命令行参数

**用法：**

```bash
# 1. 编辑文件
nano example_usage.py

# 修改这些配置：
INPUT_DIR = Path("./images")
OUTPUT_DIR = Path("./masks")
WATERMARK_REGIONS = [
    (0.8, 0.9, 1.0, 1.0),
]

# 2. 运行
python3 example_usage.py
```

### 4. `batch_remove_watermark.sh` - 一键脚本

**功能：** 从 mask 生成到批量处理的完整自动化流程

**特点：**
- ✅ 自动检查依赖
- ✅ 进度提示
- ✅ 错误处理
- ✅ 彩色输出

**配置项：**

```bash
INPUT_DIR="./images"           # 输入图片目录
MASK_DIR="./masks"             # mask 目录
OUTPUT_DIR="./output"          # 输出目录
WATERMARK_REGION="0.8 0.9 1.0 1.0"  # 水印位置
MODEL="lama"                   # 模型: lama/mat/fcf
DEVICE="cpu"                   # 设备: cpu/cuda
```

**用法：**

```bash
# 1. 编辑配置
nano batch_remove_watermark.sh

# 2. 运行
./batch_remove_watermark.sh
```

### 5. `start_iopaint.sh` - Web UI 启动器

**功能：** 启动 IOPaint Web 界面

**配置项：**
- 模型选择
- 设备选择
- 插件启用

**用法：**

```bash
# 直接启动
./start_iopaint.sh

# 访问
http://localhost:8080
```

---

## 🎯 工作流程图

```
┌─────────────────┐
│   准备图片目录   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  确定水印位置    │◄─┐
│ (visualize)     │  │
└────────┬────────┘  │
         │           │
         ▼           │
┌─────────────────┐  │
│   生成 Masks    │  │
│ (generate)      │  │
└────────┬────────┘  │
         │           │
         ▼           │
┌─────────────────┐  │
│   批量处理       │  │
│ (iopaint run)   │  │
└────────┬────────┘  │
         │           │
         ▼           │
┌─────────────────┐  │
│   检查结果       │  │
└────────┬────────┘  │
         │           │
         ▼           │
    满意? ──否───────┘
         │
         是
         ▼
     完成!
```

---

## 💻 技术架构

### 设计原则

遵循以下编码规范：
- ✅ **DRY** (Don't Repeat Yourself) - 避免重复代码
- ✅ **SOLID** - 面向对象设计原则
- ✅ **KISS** (Keep It Simple, Stupid) - 保持简单
- ✅ **高内聚，低耦合** - 模块化设计
- ✅ **关注点分离** - 清晰的职责划分

### 模块结构

```
generate_masks.py
├── MaskGenerator (核心类)
│   ├── generate_fixed_region_mask()     # 固定区域
│   ├── generate_relative_region_mask()  # 相对区域
│   ├── generate_template_matching_mask() # 模板匹配
│   └── save_mask()                      # 保存文件
├── get_image_files()                    # 文件扫描
└── batch_generate_masks()               # 批处理入口
```

### 依赖关系

```
opencv-python  ──┐
numpy          ──┼──► generate_masks.py
Pillow         ──┤
tqdm           ──┘

opencv-python  ──► visualize_watermark.py

iopaint        ──► batch_remove_watermark.sh
```

---

## 🎨 坐标系统说明

### 相对坐标（推荐）

使用 0.0 - 1.0 的相对值，自动适配不同尺寸。

```
(0,0) ──────────────── (1,0)
  │                      │
  │    (0.8, 0.9)        │
  │         ┌────────────┤
  │         │  水印区域  │
  │         │            │
(0,1) ──────┴───────── (1,1)
```

### 常用位置

| 位置 | 坐标 | 占比 |
|------|------|------|
| 右下角 | `0.8 0.9 1.0 1.0` | 20% x 10% |
| 左上角 | `0.0 0.0 0.2 0.1` | 20% x 10% |
| 底部居中 | `0.35 0.92 0.65 1.0` | 30% x 8% |

更多位置：

```bash
python3 visualize_watermark.py --list
```

---

## 📖 使用示例

### 示例1: 电商图片批量去水印

```bash
# 场景：100张产品图，右下角有商家水印

# 1. 可视化确认位置
python3 visualize_watermark.py -i products/sample.jpg \
  --region 0.82 0.88 0.98 0.98 -o check.jpg

# 2. 批量生成 mask
python3 generate_masks.py -i products -o masks \
  --region 0.82 0.88 0.98 0.98

# 3. 批量处理
export KMP_DUPLICATE_LIB_OK=TRUE
iopaint run --model=lama --device=cuda \
  --image=products --mask=masks --output=clean
```

### 示例2: 截图去除软件水印

```bash
# 场景：软件截图，左上角有软件名称

python3 generate_masks.py -i screenshots -o masks \
  --region 0.0 0.0 0.15 0.08

iopaint run --model=lama --device=cpu \
  --image=screenshots --mask=masks --output=clean
```

### 示例3: 视频截图批量处理

```bash
# 场景：视频截图，右下角时间码 + 左上角台标

python3 generate_masks.py -i frames -o masks \
  --region 0.85 0.92 1.0 1.0 \
  --region 0.0 0.0 0.12 0.08

iopaint run --model=mat --device=cuda \
  --image=frames --mask=masks --output=clean
```

---

## ⚙️ 高级配置

### 自定义模型

```bash
# 使用不同模型
iopaint run --model=mat ...      # MAT 模型
iopaint run --model=fcf ...      # FCF 模型
iopaint run --model=sd1.5 ...    # Stable Diffusion

# 查看所有可用模型
iopaint list
```

### 并行处理

```bash
# 分批处理大量图片
ls images/ | split -l 100 - batch_
# 创建多个批次，每批100张

# 分别处理
for batch in batch_*; do
    python3 generate_masks.py -i $batch -o masks_$batch \
      --region 0.8 0.9 1.0 1.0
    iopaint run --model=lama --device=cuda \
      --image=$batch --mask=masks_$batch --output=output_$batch &
done
wait
```

### 模板匹配进阶

```bash
# 1. 从图片中提取水印模板
# 使用图片编辑器裁剪出水印部分，保存为 watermark.png

# 2. 批量匹配
python3 generate_masks.py -i ./images -o ./masks \
  --template watermark.png

# 调整阈值（在代码中）
# threshold=0.8  # 默认值
# threshold=0.9  # 更严格匹配
# threshold=0.7  # 更宽松匹配
```

---

## 🔧 故障排除

### 问题1: OpenMP 库冲突 (macOS)

**症状：** `OMP: Error #15`

**解决：**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
# 或添加到 ~/.zshrc
```

### 问题2: Mask 位置不准确

**解决：**
```bash
# 使用可视化工具调整
python3 visualize_watermark.py -i sample.jpg \
  --region X1 Y1 X2 Y2 -o check.jpg
# 查看 check.jpg，调整坐标后重试
```

### 问题3: 内存不足

**解决：**
```bash
# 分批处理，减小每批数量
# 或使用 --low-mem 参数
iopaint run --model=lama --device=cpu --low-mem \
  --image=./images --mask=./masks --output=./output
```

### 问题4: 处理效果不理想

**解决：**
1. 调整 mask 区域大小（扩大或缩小）
2. 更换模型（lama/mat/fcf）
3. 使用 Web UI 手动微调
4. 增加 mask 羽化边缘

---

## 📊 性能参考

测试环境：
- CPU: Apple M1
- 图片: 1920x1080 JPG
- 模型: lama

| 操作 | 时间 | 备注 |
|------|------|------|
| 生成 mask (100张) | ~3秒 | 纯 OpenCV 操作 |
| 处理图片 (100张, CPU) | ~5分钟 | 约3秒/张 |
| 处理图片 (100张, CUDA) | ~1分钟 | 约0.6秒/张 |

---

## 🎓 教程资源

### 快速教程
1. `QUICK_START.md` - 3分钟快速上手
2. 本文档 - 工具总览和参考

### 完整教程
1. `WATERMARK_REMOVAL_GUIDE.md` - 详细使用手册
2. 各工具的 `--help` 选项

### 在线资源
- [IOPaint 官网](https://www.iopaint.com/)
- [IOPaint GitHub](https://github.com/Sanster/IOPaint)

---

## 🤝 贡献指南

欢迎提交：
- 🐛 Bug 报告
- ✨ 功能建议
- 📝 文档改进
- 💻 代码优化

---

## 📄 许可证

遵循 IOPaint 项目许可证。

---

## 📮 联系方式

遇到问题？
1. 查看 `WATERMARK_REMOVAL_GUIDE.md`
2. 检查日志输出
3. 使用 Web UI 尝试手动处理
4. 提交 Issue

---

**工具集版本：** 1.0.0  
**最后更新：** 2025-11-27  
**兼容性：** IOPaint 1.6.0+

---

**开始使用：** 查看 `QUICK_START.md` 快速上手！ 🚀



