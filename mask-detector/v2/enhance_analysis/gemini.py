import cv2
import numpy as np

def extract_watermark(image_path, output_path):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("未找到图片")
        return

    # 2. 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE (限制对比度自适应直方图均衡化)
    # 这一步非常关键！它能在不放大背景噪声太多的情况下，提升由于雾气导致的低对比度文字
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # 4. 算法核心：背景估算与差分
    # 逻辑：文字是细小的，背景是连续的。
    # 使用大核中值模糊（Median Blur）来模拟“没有文字的背景”
    # 核大小必须大于文字笔画的宽度，一般取 25-35 之间的奇数
    bg_blur = cv2.medianBlur(enhanced_gray, 31)

    # 5. 差分计算：|原图 - 背景|
    # 这会消除背景（因为背景减背景≈0），只留下差异（文字）
    # 使用 absdiff 绝对差值，无论文字比背景深还是浅，都能提取出来
    diff = cv2.absdiff(enhanced_gray, bg_blur)

    # 6. 增强差分结果的对比度 (归一化到 0-255)
    diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 7. 二值化 (Thresholding)
    # 使用 OTSU 算法自动寻找最佳阈值，或者手动调节 threshold 值
    # 这里稍微调高一点阈值以去除背景的微弱噪点
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 8. (可选) 形态学去噪
    # 如果背景中还是有很多小噪点，可以使用“开运算”去除
    kernel = np.ones((2,2), np.uint8)
    clean_watermark = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 9. 保存结果 (黑底白字)
    cv2.imwrite(output_path, clean_watermark)
    print(f"提取完成，已保存至: {output_path}")

    # 显示对比 (可选，需要图形界面支持)
    # cv2.imshow('Original', gray)
    # cv2.imshow('Enhanced', enhanced_gray)
    # cv2.imshow('Difference', diff)
    # cv2.imshow('Result', clean_watermark)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 运行提取
extract_watermark('1_x700.JPG', 'extracted_text.png')