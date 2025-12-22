#!/bin/bash

# 解决 macOS 上 OpenMP 库冲突问题
export KMP_DUPLICATE_LIB_OK=TRUE

# 启动 IOPaint 服务
# 可以根据需要修改参数
iopaint start \
  --model=lama \
  --device=cpu \
  --port=8080 \
  --enable-interactive-seg \
  --interactive-seg-device=cpu \
  --interactive-seg-model=sam2_1_tiny \
  --inbrowser

# 其他可选参数示例：
# --enable-remove-bg \
# --remove-bg-device=cpu \
# --enable-realesrgan \
# --realesrgan-device=cpu

