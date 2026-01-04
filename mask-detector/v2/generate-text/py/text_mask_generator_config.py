from PIL import Image, ImageDraw, ImageFont
import math

def add_tiled_watermark(
    input_path,
    output_path,
    text="混吸 app 1000万",
    font_path="simhei.ttf",  # 常用中文字体：黑体（Windows自带），或替换为其他ttf字体
    font_size=48,
    angle=30,                # 旋转角度（度），原图大约30度左右
    color=(255, 255, 255, 80),  # RGBA，白色半透明（原图透明度约30-40%）
    spacing=180,             # 水印之间垂直间距（可调整密度）
    horizontal_offset=200,   # 水平方向重复间隔
    vertical_offset=80       # 额外垂直偏移，使水印错位排列更自然
):
    # 打开原图
    base = Image.open(input_path).convert("RGBA")
    
    # 创建一个透明图层用于绘制水印
    watermark_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)
    
    # 加载字体（如果字体路径不存在，可换成系统其他中文字体）
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 备用字体（Pillow内置）
        font = ImageFont.load_default()
        print("指定字体未找到，使用默认字体")

    # 计算文字宽高（近似）
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 转换为弧度
    rad = math.radians(angle)
    
    # 计算旋转后需要的步进距离（使水印整齐排列）
    step_x = horizontal_offset
    step_y = spacing

    # 计算图像对角线长度，确保覆盖整个图片
    diag = math.sqrt(base.width ** 2 + base.height ** 2)
    steps = int(diag / min(step_x, step_y)) + 2

    # 在足够大的范围内绘制旋转文字
    for i in range(-steps, steps):
        for j in range(-steps, steps):
            # 计算未旋转前的中心位置
            x = i * step_x + j * vertical_offset
            y = j * step_y
            
            # 旋转坐标（围绕图片中心旋转）
            center_x = base.width / 2
            center_y = base.height / 2
            rotated_x = center_x + (x - center_x) * math.cos(rad) - (y - center_y) * math.sin(rad)
            rotated_y = center_y + (x - center_x) * math.sin(rad) + (y - center_y) * math.cos(rad)
            
            # 文字左上角位置（旋转后需要偏移半个文字大小）
            pos_x = rotated_x - text_width / 2
            pos_y = rotated_y - text_height / 2
            
            draw.text((pos_x, pos_y), text, font=font, fill=color, align="center")

    # 将水印层合并到原图
    result = Image.alpha_composite(base, watermark_layer)
    result.convert(base.mode).save(output_path, quality=95)

# ================== 使用示例 ==================
if __name__ == "__main__":
    add_tiled_watermark(
        input_path="input.jpg",           # 你的原图路径
        output_path="output_with_watermark.jpg",  # 输出路径
        text="混吸 app 1000万",           # 水印文字（可随意修改）
        font_path="simhei.ttf",           # 推荐：黑体（SimHei），或更换为其他中文字体如 msyh.ttc
        font_size=48,                     # 字号（原图约48pt）
        angle=30,                         # 倾斜角度（原图约30°）
        color=(255, 255, 255, 80),        # 白色，透明度约31%（80/255）
        spacing=180,                      # 垂直间距
        horizontal_offset=200,            # 水平重复距离
        vertical_offset=80                # 错位偏移，使排列更自然
    )
    print("水印添加完成！")