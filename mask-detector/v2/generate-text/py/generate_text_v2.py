from PIL import Image, ImageDraw, ImageFont

def create_watermark_mask(input_path, output_path, config):
    base_img = Image.open(input_path)
    w, h = base_img.size

    # 创建黑底画布
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # 加载字体
    f1 = ImageFont.truetype(config['font_path'], config['line1']['size'])
    f2_bold = ImageFont.truetype(config['font_bold_path'], config['line2'][0]['size'])
    f2_reg = ImageFont.truetype(config['font_path'], config['line2'][1]['size'])

    # 计算文字宽度
    line1_text = config['line1']['text']
    line1_w = draw.textlength(line1_text, font=f1)
    
    line2_part1 = config['line2'][0]['text']
    line2_part2 = config['line2'][1]['text']
    line2_w1 = draw.textlength(line2_part1, font=f2_bold)
    line2_w2 = draw.textlength(line2_part2, font=f2_reg)
    line2_w = line2_w1 + config['line2'][0]['margin_r'] + line2_w2

    # 间距
    gap1 = config['line1']['gap']  # 第一行重复间距
    gap2 = config['line2_gap']      # 第二行重复间距
    row_gap = config['row_gap']     # 行与行间距

    # 绘制：每行独立水平重复填充
    y = 0
    row_idx = 0
    while y < h:
        if row_idx % 2 == 0:
            # 第一行：雪票、酒店...
            x = 0
            while x < w:
                draw.text((x, y), line1_text, font=f1, fill=255)
                x += line1_w + gap1
            y += int(f1.getbbox(line1_text)[3]) + row_gap
        else:
            # 第二行：滑呗 app + 1000万...
            x = 0
            while x < w:
                draw.text((x, y), line2_part1, font=f2_bold, fill=255)
                x2 = x + line2_w1 + config['line2'][0]['margin_r']
                draw.text((x2, y + 8), line2_part2, font=f2_reg, fill=255)  # +8 视觉对齐
                x += line2_w + gap2
            y += int(f2_bold.getbbox(line2_part1)[3]) + row_gap
        row_idx += 1

    # 暂时不旋转
    rotated = mask.rotate(config['angle'], resample=Image.BICUBIC)
    rotated.save(output_path, "PNG")
    # mask.save(output_path, "PNG")
    print(f"✓ Mask已生成: {output_path}")

# --- 配置区 ---
watermark_config = {
    "font_path": "/System/Library/Fonts/STHeiti Light.ttc",
    "font_bold_path": "/System/Library/Fonts/STHeiti Medium.ttc",
    "angle": 18,
    "row_gap": 20,       # 行与行间距
    "line1": {
        "text": "雪票、酒店、教练、摄影师、约玩",
        "size": 16,
        "gap": 80,       # 第一行重复间距
    },
    "line2": [
        {"text": "滑呗app", "size": 36, "margin_r": 4},
        {"text": "1000万雪友的选择", "size": 22}
    ],
    "line2_gap": 100,    # 第二行重复间距
}

if __name__ == "__main__":
    create_watermark_mask("input.jpg", "final_result.png", watermark_config)
