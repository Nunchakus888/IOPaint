
### 实现思路

1. **创建透明图层**：文字不能直接画在原图上，否则难以控制透明度。我们需要新建一个和原图一样大的透明图层。
2. **计算平铺间距**：为了让文字铺满，我们需要用循环来控制 X 和 Y 轴的坐标。
3. **旋转与绘制**：使用 `ImageDraw` 绘制文字，并利用 `Image.rotate` 或在绘制前对画布进行坐标变换。
4. **合并图层**：将文字图层通过 `alpha_composite` 叠加到原图。

### Python 代码实现

你需要先安装 Pillow：`pip install Pillow`

```python
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def create_watermark(input_image_path, output_image_path, text, font_path, font_size=40, color=(255, 255, 255, 80), angle=30, spacing=150):
    """
    :param input_image_path: 原图路径
    :param output_image_path: 输出路径
    :param text: 水印文字
    :param font_path: 字体文件路径 (如 simsun.ttc)
    :param font_size: 字体大小
    :param color: 文字颜色和透明度 (R, G, B, Alpha)
    :param angle: 旋转角度
    :param spacing: 文字之间的间距
    """
    # 1. 打开原图并转为 RGBA
    base_img = Image.open(input_image_path).convert("RGBA")
    width, height = base_img.size

    # 2. 创建一个透明的图层用于画文字
    # 为了旋转后不留白边，我们创建一个比原图大一倍的临时图层
    txt_layer = Image.new("RGBA", (width * 2, height * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)
    
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print("字体文件未找到，将使用默认字体")
        font = ImageFont.load_default()

    # 3. 在大图层上平铺文字
    # 根据间距循环绘制
    for x in range(0, width * 2, spacing * 2):
        for y in range(0, height * 2, spacing):
            draw.text((x, y), text, font=font, fill=color)

    # 4. 旋转文字图层
    rotated_txt = txt_layer.rotate(angle, expand=False)

    # 5. 裁剪并合并
    # 从旋转后的大图中截取原图大小的部分（取中心部分）
    crop_x = (rotated_txt.width - width) // 2
    crop_y = (rotated_txt.height - height) // 2
    final_txt_layer = rotated_txt.crop((crop_x, crop_y, crop_x + width, crop_y + height))

    # 合成图片
    out = Image.alpha_composite(base_img, final_txt_layer)
    
    # 转回 RGB 并保存
    out.convert("RGB").save(output_image_path, "PNG")
    print(f"成功生成！已保存至: {output_image_path}")

# --- 参数配置 ---
config = {
    "input_image_path": "04_sobel.png", # 你的原图
    "output_image_path": "output_watermarked.png",
    "text": "滑呗app1000万雪友的选择", 
    "font_path": "msyh.ttc",            # 微软雅黑或其他系统中文字体
    "font_size": 24,                    # 字体大小
    "color": (255, 255, 255, 60),       # 白色，60是透明度(0-255)
    "angle": 30,                        # 倾斜 30 度
    "spacing": 180                      # 密度，数值越大越稀疏
}

if __name__ == "__main__":
    create_watermark(**config)

```

---

### 核心调优建议

* **字体 (Font)**：图中看起来像是“微软雅黑”或者类似的黑体。如果你在 Windows 上，`font_path` 可以写 `C:/Windows/Fonts/msyh.ttc`。
* **透明度 (Alpha)**：图中水印比较淡。在 `color` 参数的第四位控制（如 `60`），数值越低越透明。
* **错位排列**：如果想要实现图中那种每行交错的效果，可以在循环里加一个偏移：
```python
for i, y in enumerate(range(0, height * 2, spacing)):
    offset = spacing if i % 2 == 0 else 0  # 奇偶行偏移
    for x in range(0, width * 2, spacing * 2):
        draw.text((x + offset, y), text, font=font, fill=color)

```



### 下班指南

1. 把代码里的 `text` 改成老板要的内容。
2. 调整 `font_size` 和 `spacing` 直到视觉效果最接近。
3. 运行脚本，一秒批量生成。

需要我帮你针对具体的文字内容或者更复杂的交错排版再优化一下代码吗？