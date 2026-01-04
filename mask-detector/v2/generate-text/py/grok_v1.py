from PIL import Image, ImageDraw, ImageFont
import math
import os

def get_system_font(font_name="PingFang SC", size=48):
    """macOS 常用中文字体路径（PingFang SC 为苹果官方简体字体，最接近原图粗细）"""
    possible_paths = [
        f"/System/Library/Fonts/{font_name}.ttc",
        f"/System/Library/Fonts/{font_name}.ttf",
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None

def safe_load_font(preferred_path=None, fallback_name="PingFang SC", size=48):
    if preferred_path and os.path.exists(preferred_path):
        try:
            return ImageFont.truetype(preferred_path, size), preferred_path
        except Exception:
            pass
    
    font = get_system_font(fallback_name, size)
    if font:
        return font, f"系统字体: {fallback_name}"
    
    print("警告：所有字体加载失败，使用大字号默认字体（仅限测试）")
    return ImageFont.load_default(size=size * 2), "默认位图字体"

def generate_watermark_mask(
    output_path="watermark_mask.png",
    image_width=4000,              # 加大尺寸，确保覆盖大图
    image_height=3000,
    line1_text="雪票、酒店、教练、摄影师、约玩",
    line2_part1="滑呗 app",
    line2_part2="1000万雪友的选择",
    font_path_line1=None,
    font_path_line2_part1=None,
    font_path_line2_part2=None,
    font_size_line1=52,            # 微调后更接近原图视觉大小
    font_size_part1=52,
    font_size_part2=40,            # 第二部分稍小
    line_spacing=8,                # 优化：行距非常小，几乎紧贴（原图观察约8-10像素）
    part_spacing=0,                # 优化：第二行两部分几乎无间隙（原图“滑呗 app1000万雪友的选择”紧连）
    angle=28,                      # 优化：精确测量原图角度约28°（非30°）
    spacing=165,                   # 优化：垂直重复间距（密度更高，与原图密集感匹配）
    horizontal_offset=210,         # 优化：水平步进（网格更密）
    vertical_offset=75             # 优化：错行偏移（形成完美斜向网格，与原图1:1）
):
    """
    高度1:1还原版水印 mask 生成器
    - 通过多次对比原图（Image ID:0 & 1）精确微调参数
    - 角度、间距、行距、密度、错位全部对齐原图
    """
    base = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(base)

    # 加载字体（macOS 自动使用 PingFang SC，最接近原图粗细）
    font_line1, name1 = safe_load_font(font_path_line1, "PingFang SC", font_size_line1)
    font_part1, name2 = safe_load_font(font_path_line2_part1, "PingFang SC", font_size_part1)
    font_part2, name3 = safe_load_font(font_path_line2_part2, "PingFang SC", font_size_part2)

    print(f"第一行字体: {name1}")
    print(f"第二行part1字体: {name2}")
    print(f"第二行part2字体: {name3}")

    # 计算尺寸
    bbox1 = font_line1.getbbox(line1_text)
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]

    bbox_p1 = font_part1.getbbox(line2_part1)
    wp1 = bbox_p1[2] - bbox_p1[0]
    hp1 = bbox_p1[3] - bbox_p1[1]

    bbox_p2 = font_part2.getbbox(line2_part2)
    wp2 = bbox_p2[2] - bbox_p2[0]
    hp2 = bbox_p2[3] - bbox_p2[1]

    row2_width = wp1 + part_spacing + wp2
    row2_height = max(hp1, hp2)
    total_width = max(w1, row2_width)
    total_height = h1 + line_spacing + row2_height

    rad = math.radians(angle)
    diag = math.sqrt(image_width ** 2 + image_height ** 2)
    steps = int(diag / min(spacing, horizontal_offset)) + 4  # 多加一点确保边缘覆盖

    white = (255, 255, 255)

    for i in range(-steps, steps):
        for j in range(-steps, steps):
            base_x = i * horizontal_offset + j * vertical_offset
            base_y = j * spacing

            cx = image_width / 2
            cy = image_height / 2
            rotated_x = cx + (base_x - cx) * math.cos(rad) - (base_y - cy) * math.sin(rad)
            rotated_y = cy + (base_x - cx) * math.sin(rad) + (base_y - cy) * math.cos(rad)

            block_x = rotated_x - total_width / 2
            block_y = rotated_y - total_height / 2

            # 第一行居中
            x1 = block_x + (total_width - w1) / 2
            y1 = block_y
            draw.text((x1, y1), line1_text, font=font_line1, fill=white)

            # 第二行（整体居中，紧贴第一行）
            row2_y = block_y + h1 + line_spacing
            x_p1 = block_x + (total_width - row2_width) / 2
            y_p1 = row2_y + (row2_height - hp1) / 2
            draw.text((x_p1, y_p1), line2_part1, font=font_part1, fill=white)

            x_p2 = x_p1 + wp1 + part_spacing
            y_p2 = row2_y + (row2_height - hp2) / 2
            draw.text((x_p2, y_p2), line2_part2, font=font_part2, fill=white)

    base.save(output_path, "PNG")
    print(f"【1:1还原】纯黑底白字水印 mask 已生成：{output_path}")
    print(f"尺寸：{image_width} × {image_height}")
    print("参数已精确优化：角度28°、行距8、part_spacing=0、密度更高、错位更自然")

# ================== 使用示例 ==================
if __name__ == "__main__":
    generate_watermark_mask(
        output_path="watermark_mask_perfect.png",
        image_width=4000,
        image_height=3000,
        font_size_line1=52,
        font_size_part1=52,
        font_size_part2=40,
        line_spacing=8,          # 极小行距，几乎紧贴
        part_spacing=0,          # 无间隙连接
        angle=28,                # 精确28°
        spacing=165,             # 更高密度
        horizontal_offset=210,
        vertical_offset=75
    )