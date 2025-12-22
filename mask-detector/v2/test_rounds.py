#!/usr/bin/env python3
"""
批量测试不同轮次的优化版水印检测器
"""

import os
import subprocess
import sys

def test_round(round_num):
    """测试指定轮次"""
    print(f"\n🧪 测试轮次 {round_num}")

    # 创建目录和输入文件
    round_dir = str(round_num)
    os.makedirs(round_dir, exist_ok=True)

    input_path = os.path.join(round_dir, 'input.jpg')
    if not os.path.exists(input_path):
        # 使用示例图像
        sample_path = '../../../../images/sample.jpg'
        if os.path.exists(sample_path):
            os.system(f'cp {sample_path} {input_path}')
            print(f"  📋 复制输入图像: {input_path}")
        else:
            print(f"  ⚠️ 跳过轮次 {round_num}：找不到输入图像")
            return False

    # 运行检测器
    cmd = [sys.executable, 'watermark_detector_optimized.py', '-r', str(round_num), '--preview']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✅ 轮次 {round_num} 处理成功")

        # 检查输出文件
        mask_path = os.path.join(round_dir, 'mask.png')
        preview_path = os.path.join(round_dir, 'detection_preview.jpg')

        if os.path.exists(mask_path):
            print(f"  💾 Mask: {mask_path}")
        if os.path.exists(preview_path):
            print(f"  🖼️ Preview: {preview_path}")

        return True
    else:
        print(f"  ❌ 轮次 {round_num} 处理失败")
        print(f"  错误信息: {result.stderr}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始批量测试优化版水印检测器")

    # 测试多个轮次
    test_rounds = [1, 2, 3]
    success_count = 0

    for round_num in test_rounds:
        if test_round(round_num):
            success_count += 1

    print(f"\n📊 测试结果: {success_count}/{len(test_rounds)} 轮次成功")

    if success_count > 0:
        print("\n📁 生成的文件结构:")
        for round_num in test_rounds:
            round_dir = str(round_num)
            if os.path.exists(round_dir):
                files = os.listdir(round_dir)
                print(f"  轮次 {round_num}: {files}")

    print("\n💡 使用提示:")
    print("  - 如果安装了IOPaint，会自动进行水印去除")
    print("  - 未安装时会显示手动命令格式")
    print("  - 可以单独运行去除命令进行水印清理")

    if success_count == len(test_rounds):
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 部分测试失败，请检查配置")

if __name__ == "__main__":
    main()
