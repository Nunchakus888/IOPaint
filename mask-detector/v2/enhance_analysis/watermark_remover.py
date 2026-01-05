"""水印去除模块"""
import os
import subprocess
import cv2
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_image_size(path: str) -> Optional[tuple]:
    """获取图片尺寸 (w, h)"""
    img = cv2.imread(path)
    return (img.shape[1], img.shape[0]) if img is not None else None


def load_masks(mask_dir: str) -> Dict[str, str]:
    """加载 mask 目录下的所有 PNG 文件"""
    if not os.path.exists(mask_dir):
        return {}
    return {
        f: os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir) if f.endswith('.png')
    }


def remove_watermark(
    img_path: str,
    mask_path: str,
    output_dir: str,
    model: str = 'lama',
    device: str = 'cpu',
    verbose: bool = False
) -> Optional[str]:
    """
    使用 iopaint 去除单张图片水印
    
    Args:
        model: 支持 lama(快速) / mat(质量更高) / cv2 / zits / ldm 等
        verbose: 是否显示详细输出（首次下载模型时建议开启）
    
    Returns:
        输出文件路径，失败返回 None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'iopaint', 'run',
        f'--model={model}',
        f'--device={device}',
        f'--image={img_path}',
        f'--mask={mask_path}',
        f'--output={output_dir}'
    ]
    
    # verbose=True 时显示输出（模型下载进度等）
    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode != 0:
        return None
    
    # 返回输出文件路径
    base_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.png'
    return os.path.join(output_dir, base_name)


def refine_with_mat(
    img_path: str,
    mask_path: str,
    output_path: str,
    device: str = 'cpu',
    verbose: bool = True
) -> bool:
    """
    使用 MAT 模型进行二次修复（处理残留）
    
    MAT 基于 Transformer，纹理生成能力更强，适合处理残留痕迹
    首次使用会下载模型（约200MB），建议 verbose=True 查看进度
    """
    output_dir = os.path.dirname(output_path)
    result = remove_watermark(
        img_path, mask_path, output_dir, 
        model='mat', device=device, verbose=verbose
    )
    
    if result and os.path.exists(result):
        # 如果输出文件名不同，重命名
        if result != output_path:
            os.replace(result, output_path)
        return True
    return False


def batch_remove_watermarks(
    images: List[str],
    input_dir: str,
    mask_dir: str,
    output_dir: str,
    model: str = 'lama',
    device: str = 'cpu',
    on_progress: Callable = None,
    max_workers: int = 1
) -> List[str]:
    """
    批量去除水印（支持并发）
    
    Args:
        max_workers: 并发数，1 为串行，>1 为并发
    """
    masks = load_masks(mask_dir)
    
    # 构建任务列表
    tasks = []
    for img_path in images:
        size = get_image_size(img_path)
        if not size:
            continue
        mask_name = f"mask_{size[0]}x{size[1]}.png"
        if mask_name not in masks:
            continue
        rel_path = os.path.relpath(img_path, input_dir)
        out_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        tasks.append((img_path, masks[mask_name], out_subdir))
    
    results = []
    
    def process_one(task):
        img_path, mask_path, out_subdir = task
        return img_path, remove_watermark(img_path, mask_path, out_subdir, model, device)
    
    # 串行或并发执行
    if max_workers <= 1:
        for task in tasks:
            img_path, output_path = process_one(task)
            if output_path:
                results.append(output_path)
                if on_progress:
                    on_progress(img_path, output_path)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one, t): t for t in tasks}
            for future in as_completed(futures):
                img_path, output_path = future.result()
                if output_path:
                    results.append(output_path)
                    if on_progress:
                        on_progress(img_path, output_path)
    
    return results

