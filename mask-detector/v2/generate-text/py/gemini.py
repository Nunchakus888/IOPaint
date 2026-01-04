import math
import os
from PIL import Image, ImageDraw, ImageFont

# ================= 1. 配置区域 =================

CONFIG = {
    # 基础设置
    "image_path": "input.jpg",       # 输入图路径
    "output_path": "output_restored_fixed.jpg", # 输出结果
    "canvas_bg": "black",            # 演示用底色
    
    # 全局样式
    "angle": 25,                     # 旋转角度
    "opacity": 1,                 # 水印透明度
    "row_spacing": 90,              # 行与行之间的垂直距离
    "item_spacing": 60,              # 同一行内文字块的水平间距
    
    # === 行内容定义 ===
    "rows": [
        # --- 第一行：小字描述 ---
        [
            {
                "text": "雪票、酒店、教练、摄影师、约玩",
                "size": 24,
                "bold": False,
                "offset_y": 0
            }
        ],
        
        # --- 第二行：品牌大字 + Slogan ---
        [
            {
                "text": "滑呗",
                "size": 28,
                "bold": True,  # 使用粗体字体
                "offset_y": 0
            },
            {
                "text": "app 1000万雪友的选择",
                "size": 24,
                "bold": False,
                "offset_y": 0
            }
        ]
    ]
}

# ================= 2. 渲染引擎 (修复版) =================

class WatermarkRenderer:
    def __init__(self, config):
        self.cfg = config
        self.font_cache = {}

    def get_font(self, size, bold=False):
        """
        字体加载逻辑，支持粗体。
        macOS PingFang.ttc 中不同 index 对应不同字重：
        - index 0: Regular
        - index 1: Medium
        - index 2: Semibold
        - index 3: Bold (如果存在)
        """
        cache_key = (size, bold)
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        # macOS 常见中文字体列表，配置 (路径, 常规index, 粗体index)
        potential_fonts = [
            # PingFang: index 2 是 Semibold，比较接近粗体效果
            ("/System/Library/Fonts/PingFang.ttc", 0, 2),
            # STHeiti Medium 本身就是中等粗细
            ("/System/Library/Fonts/STHeiti Medium.ttc", 0, 0),
            ("/System/Library/Fonts/STHeiti Light.ttc", 0, 0),
            ("/Library/Fonts/Arial Unicode.ttf", 0, 0),
        ]

        font = None
        loaded_path = ""

        for font_path, regular_idx, bold_idx in potential_fonts:
            if os.path.exists(font_path):
                try:
                    idx = bold_idx if bold else regular_idx
                    font = ImageFont.truetype(font_path, size, index=idx)
                    loaded_path = font_path
                    break
                except Exception:
                    continue
        
        # 如果系统字体都加载失败，尝试加载当前目录下的自定义字体
        if font is None:
            local_font = "simhei.ttf"
            if os.path.exists(local_font):
                try:
                    font = ImageFont.truetype(local_font, size)
                    loaded_path = local_font
                except:
                    pass

        if font is None:
            raise RuntimeError(
                "\n❌ 严重错误：未找到支持中文的系统字体。\n"
                "请下载一个中文字体文件(如 simhei.ttf 或msyh.ttf)，\n"
                "放在代码同级目录下，并修改代码中的 font_path。\n"
            )

        if cache_key not in self.font_cache:
            weight = "Bold" if bold else "Regular"
            print(f"✅ 已加载字体: {loaded_path} (Size: {size}, {weight})")
        
        self.font_cache[cache_key] = font
        return font

    def measure_segment(self, draw, segment):
        font = self.get_font(segment["size"], segment.get("bold", False))
        bbox = draw.textbbox((0, 0), segment["text"], font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ascent, descent = font.getmetrics()
        return width, height, ascent, font

    def draw_composite_line(self, draw, start_x, start_y, segments):
        # 1. 第一遍扫描：找出这一行中最高的基线位置
        max_ascent = 0
        seg_measurements = []
        
        for seg in segments:
            w, h, ascent, font = self.measure_segment(draw, seg)
            if ascent > max_ascent:
                max_ascent = ascent
            seg_measurements.append((w, ascent, font))

        # 2. 绘制
        current_x = start_x
        total_width = 0
        
        fill_rgba = (0, 0, 0, 255)  # 黑色实心文字

        for i, seg in enumerate(segments):
            w, ascent, font = seg_measurements[i]
            
            # 基线对齐核心逻辑
            draw_y = start_y + (max_ascent - ascent) + seg.get("offset_y", 0)

            draw.text(
                (current_x, draw_y),
                seg["text"],
                font=font,
                fill=fill_rgba
            )
            
            current_x += w
            total_width += w
            
        return total_width, max_ascent

    def run(self):
        # 1. 准备底图
        if os.path.exists(self.cfg["image_path"]):
            img = Image.open(self.cfg["image_path"]).convert("RGBA")
        else:
            print(f"提示：未找到 {self.cfg['image_path']}，生成黑色背景演示。")
            img = Image.new("RGBA", (2000, 1500), self.cfg["canvas_bg"])
        
        W, H = img.size
        diag = int(math.sqrt(W**2 + H**2)) * 1.5
        canvas_side = int(diag)
        
        txt_layer = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        cx, cy = canvas_side // 2, canvas_side // 2
        y = -canvas_side // 2
        row_index = 0
        
        while y < canvas_side:
            row_config = self.cfg["rows"][row_index % len(self.cfg["rows"])]
            
            dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
            line_w, line_h_ascent = self.draw_composite_line(dummy_draw, 0, 0, row_config)
            
            true_line_height = line_h_ascent * 1.3 
            full_block_w = line_w + self.cfg["item_spacing"]
            
            start_x = -(canvas_side // 2) - full_block_w
            offset_x = (full_block_w * 0.5) if row_index % 2 == 1 else 0
            
            x = start_x + offset_x
            while x < canvas_side:
                self.draw_composite_line(draw, x + cx, y + cy, row_config)
                x += full_block_w
            
            y += true_line_height + self.cfg["row_spacing"]
            row_index += 1

        rotated_layer = txt_layer.rotate(self.cfg["angle"], resample=Image.BICUBIC)
        
        left = (rotated_layer.width - W) // 2
        top = (rotated_layer.height - H) // 2
        crop_layer = rotated_layer.crop((left, top, left + W, top + H))
        
        if self.cfg["opacity"] < 1.0:
            r, g, b, a = crop_layer.split()
            a = a.point(lambda i: int(i * self.cfg["opacity"]))
            crop_layer = Image.merge("RGBA", (r, g, b, a))

        result = Image.alpha_composite(img, crop_layer)
        result = result.convert("RGB")
        result.save(self.cfg["output_path"], quality=95)
        print(f"✅ 生成完毕: {self.cfg['output_path']}")

if __name__ == "__main__":
    try:
        renderer = WatermarkRenderer(CONFIG)
        renderer.run()
    except Exception as e:
        print(e)