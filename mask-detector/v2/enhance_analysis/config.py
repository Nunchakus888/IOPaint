"""统一配置管理"""
import os

# === 路径配置 ===
INPUT_DIR = 'enhance_analysis/images/2025-11-22'
MASK_DIR = 'enhance_analysis/masks'
OUTPUT_DIR = 'enhance_analysis/output/2025-11-22/0'

# === 模型配置 ===
# 支持模型: lama, mat, cv2, zits, ldm, fcf, manga, migan
# 推荐: lama (快速) / mat (质量更高)
INPAINT_MODEL = 'lama'
DEVICE = 'cpu'

# === 功能开关 ===
ENABLE_ENHANCE = True   # 是否启用 RealESRGAN 增强
ENABLE_REFINE = False   # 是否启用二次修复（用MAT处理残留）
VERBOSE = True          # 显示详细输出（首次下载模型时必须开启）
MAX_WORKERS = 2         # 并发数（⚠️ 建议：CPU核心数/2，为系统保留资源）

# === 环境设置 ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 限制 PyTorch/OpenMP 线程数，避免资源耗尽
# 每个worker使用1-2个线程，总线程数 = MAX_WORKERS * THREADS_PER_WORKER
THREADS_PER_WORKER = 2
os.environ['OMP_NUM_THREADS'] = str(THREADS_PER_WORKER)
os.environ['MKL_NUM_THREADS'] = str(THREADS_PER_WORKER)
os.environ['NUMEXPR_NUM_THREADS'] = str(THREADS_PER_WORKER)

