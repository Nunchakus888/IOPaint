import math
import os
import sys
import urllib.request
import ssl
from PIL import Image, ImageDraw, ImageFont

# ================= 0. 自动字体下载模块 =================

# 这里使用 Adobe 官方开源的思源黑体 (Bold 字重)，文件约 16MB
FONT_URL = "https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SimplifiedChinese/SourceHanSansSC-Bold.otf"
FONT_FILENAME = "SourceHanSansSC-Bold.otf"

def check_and_download_font():
    """检查本地是否有字体，没有则自动下载"""
    if os.path.exists(FONT_FILENAME):
        return FONT_FILENAME
    
    print(f"🔍 未检测到本地字体，正在从 GitHub 下载 {FONT_FILENAME} ...")
    print("⏳ 这可能需要几秒钟到一分钟，取决于你的网速，请耐心等待...")
    
    try:
        # 处理 HTTPS 上下文 (防止 macOS SSL 报错)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        # 带有 User-Agent 伪装，防止 Github 拒绝请求
        req = urllib.request.Request(
            FONT_URL, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        
        with urllib.request.urlopen(req, context=ctx) as response, open(FONT_FILENAME, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        print(f"✅ 字体下载成功：{os.path.abspath(FONT_FILENAME)}")
        return FONT_FILENAME
        
    except Exception as e:
        print(f"\n❌ 字体下载失败: {e}")
        print(f"请手动下载此文件: {FONT_URL}")
        print(f"并重命名为 {FONT_FILENAME} 放在代码同级目录下。")
        # 如果下载失败，返回 None，后面会报错
        return None

# ================= 1. 1:1 还原配置 =================

CONFIG = {
    "image_path": "input.jpg",       
    "output_path": "output_auto_font.jpg",
    "debug": True,           # 调试模式：输出完整画布 + 半透明原图叠加
    
    # 物理参数
    "angle": 25,            # 旋转角度
    "text_x": 0,             # 文字层水平偏移（旋转前，正=右移）
    "text_y": 0,             # 文字层垂直偏移（旋转前，正=下移）
    "row_spacing": 40,       # 行间距
    "item_spacing": 40,      # 列间距
    "stagger": 0.5,            # 奇数行错位 (0=不错位, 0.5=半块宽)
    
    # 行内容（简化格式）
    "rows": [
        # 第1行
        [{"text": "滑呗", "size": 28, "stroke": 3}, {"text": "app 1000万雪友的选择", "size": 24, "stroke": 1}],
        # 第2行
        [{"text": "雪票、酒店、教练、摄影师、约玩", "size": 24, "stroke": 0}],
    ],
}

# ================= 2. 渲染引擎 =================

class WatermarkRenderer:
    def __init__(self, config):
        self.cfg = config
        self.font_cache = {}
        
        # 1. 优先使用自动下载的字体
        self.local_font_path = check_and_download_font()
        
        # 2. 备用系统字体列表 (路径, 常规Index, 粗体Index)
        self.sys_font_candidates = [
            ("/System/Library/Fonts/PingFang.ttc", 0, 2),
            ("/System/Library/Fonts/STHeiti Medium.ttc", 0, 0),
        ]

    def get_font(self, size, weight="Regular"):
        """
        字体加载逻辑：优先本地下载的OTF -> 其次系统TTC
        """
        cache_key = (size, weight)
        if cache_key in self.font_cache: 
            return self.font_cache[cache_key]
        
        font = None
        is_bold = (weight == "Bold" or weight == "Semibold")

        # 尝试 A: 本地自动下载的思源黑体
        if self.local_font_path and os.path.exists(self.local_font_path):
            try:
                # 注意：OTF 通常不支持 index参数 (除非是 OTC)，这点不同于 TTC
                # 思源黑体本身就是 Bold 版，所以无论 Regular 还是 Bold 请求都返回这个
                # 对于 Regular 需求，我们通过减小 stroke_width 来从视觉上变细
                font = ImageFont.truetype(self.local_font_path, size)
            except Exception:
                pass

        # 尝试 B: macOS 系统字体 (如果没有下载成功)
        if font is None:
            for path, reg_idx, bold_idx in self.sys_font_candidates:
                if os.path.exists(path):
                    try:
                        idx = bold_idx if is_bold else reg_idx
                        font = ImageFont.truetype(path, size, index=idx)
                        break
                    except:
                        continue
        
        # 毁灭性错误检查
        if font is None:
            raise RuntimeError("❌ 无法加载任何字体！网络下载失败且未找到系统字体。")

        self.font_cache[cache_key] = font
        return font

    def measure_segment(self, draw, seg):
        font = self.get_font(seg["size"])
        stroke = seg.get("stroke", 0)
        bbox = draw.textbbox((0, 0), seg["text"], font=font, stroke_width=stroke)
        return bbox[2] - bbox[0], font.getmetrics()[0], font

    def draw_composite_line(self, draw, x, y, segments):
        max_asc = max(self.measure_segment(draw, s)[1] for s in segments)
        start_x = x
        for seg in segments:
            w, asc, font = self.measure_segment(draw, seg)
            stroke = seg.get("stroke", 0)
            draw.text((x, y + max_asc - asc), seg["text"], font=font,
                      fill=(0, 0, 0, 255), stroke_width=stroke, stroke_fill=(255, 255, 255, 255))
            x += w
        return x - start_x, max_asc

    def run(self):
        # 底图
        if os.path.exists(self.cfg["image_path"]):
            img = Image.open(self.cfg["image_path"]).convert("RGBA")
        else:
            img = Image.new("RGBA", (1920, 1080), "black")
            
        W, H = img.size
        diag = int(math.sqrt(W**2 + H**2) * 1.5)  # 1.5倍确保旋转后覆盖中心
        
        # 文字层
        tile = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        
        rows = self.cfg["rows"]
        row_sp = self.cfg.get("row_spacing", 40)
        item_sp = self.cfg.get("item_spacing", 40)
        stagger = self.cfg.get("stagger", 0)
        angle_rad = math.radians(self.cfg["angle"])
        tan_angle = math.tan(angle_rad)
        
        # 文字层整体偏移（旋转前）
        text_x = self.cfg.get("text_x", 0)
        text_y = self.cfg.get("text_y", 0)
        
        # 预计算行尺寸
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        metrics = [self.draw_composite_line(dummy, 0, 0, r) for r in rows]
        
        # 铺满画布
        y, idx = text_y, 0
        
        while y < diag:
            row = rows[idx % len(rows)]
            lw, lh = metrics[idx % len(rows)]
            full_w = lw + item_sp
            
            # 根据 y 位置计算左偏移 = y * tan(angle)，使旋转后垂直对齐
            left_offset = y * tan_angle
            base_x = text_x - left_offset
            stagger_x = (full_w * stagger) if idx % 2 else 0
            
            x = base_x + stagger_x
            while x < diag + left_offset:
                self.draw_composite_line(draw, x, y, row)
                x += full_w
            y += lh + row_sp
            idx += 1
        
        # 旋转
        tile = tile.rotate(self.cfg["angle"], resample=Image.BICUBIC)
        
        # 裁剪位置（居中）
        left = (tile.width - W) // 2
        top = (tile.height - H) // 2
        
        # 输出
        if self.cfg.get("debug"):
            result = _debug_preview(tile, img, left, top, W, H)
        else:
            img.alpha_composite(tile.crop((left, top, left+W, top+H)))
            result = img
        
        result.convert("RGB").save(self.cfg["output_path"], quality=95)
        print(f"✨ 已生成: {self.cfg['output_path']}")


# ================= 调试模块（调试完成后可删除） =================

def _debug_preview(tile, img, left, top, W, H):
    """生成调试预览图：灰底 + 文字层 + 半透明原图 + 红框"""
    canvas = Image.new("RGBA", (tile.width, tile.height), (80, 80, 80, 255))
    canvas.alpha_composite(tile)
    img_t = img.copy()
    img_t.putalpha(100)
    canvas.paste(img_t, (left, top), img_t)
    ImageDraw.Draw(canvas).rectangle([(left, top), (left+W-1, top+H-1)], outline=(255, 0, 0), width=2)
    return canvas

# ================= 调试模块结束 =================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=int, default=None, help="文字层水平偏移 (正=右移)")
    parser.add_argument("-y", type=int, default=None, help="文字层垂直偏移 (正=下移)")
    args = parser.parse_args()
    
    # 命令行参数覆盖配置（偏移文字层，非裁剪位置）
    if args.x is not None: CONFIG["text_x"] = args.x
    if args.y is not None: CONFIG["text_y"] = args.y
    
    try:
        renderer = WatermarkRenderer(CONFIG)
        renderer.run()
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")