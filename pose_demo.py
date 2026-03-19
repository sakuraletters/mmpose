import os
import cv2
import numpy as np
import argparse
import json
from mmpose.apis import MMPoseInferencer

# ==========================================
# 1. 序列化与基础工具函数
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """
    自定义的 JSON 编码器。
    用于在执行 json.dump() 时拦截并转换 NumPy 特有的数据类型。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_image_paths(input_dir: str) -> list:
    """递归遍历输入目录，获取所有常见格式的图像物理路径。"""
    if os.path.isfile(input_dir):
        return [input_dir]
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
                
    return sorted(image_paths)

def make_transparent_png(bgr_image: np.ndarray) -> np.ndarray:
    """将黑底 BGR 图像矩阵转换为透明背景的 BGRA PNG 矩阵。"""
    # dtype 保险：MMPose 某些配置下返回 float32，必须转为 uint8 才能正确写 PNG
    if bgr_image.dtype != np.uint8:
        bgr_image = (bgr_image * 255).clip(0, 255).astype(np.uint8)
    b, g, r = cv2.split(bgr_image)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    black_mask = (b == 0) & (g == 0) & (r == 0)
    alpha[black_mask] = 0
    bgra_image = cv2.merge((b, g, r, alpha))
    return bgra_image

# ==========================================
# 2. 核心特征处理：直接内存运算替代磁盘读取
# ==========================================
def extract_and_save_txt(predictions: list, txt_path: str, target_size: float = 128.0):
    """
    直接从内存中的字典提取主目标，并将绝对坐标归一化到指定尺寸的画布中。
    （由原 j2t.py 改写，去除了文件读取操作以提升性能）
    """
    # 1. 基础校验
    if not isinstance(predictions, list) or len(predictions) == 0:
        print(f"警告：未检测到任何目标，跳过 txt 提取 -> {txt_path}")
        return

    # 2. 核心修复：处理 MMPose 的批次嵌套问题
    # 如果 predictions 的第一个元素是个列表，说明它被嵌套成了 [[{...}, {...}]]
    if isinstance(predictions[0], list):
        predictions = predictions[0]  # 提取内层真正的实例列表

    # 解包后再次校验是否为空（防止出现 [ [] ] 的情况）
    if len(predictions) == 0:
        return

    # 3. 提取置信度最高的主目标 (此时 predictions 已经是纯粹的字典列表 [{...}, {...}])
    best_instance = max(predictions, key=lambda x: x.get('bbox_score', 0))
    
    keypoints = best_instance.get('keypoints', [])
    scores = best_instance.get('keypoint_scores', [])
    bbox_list = best_instance.get('bbox', [])
    
    if len(keypoints) == 0 or len(bbox_list) == 0:
        return
        
    # 处理内存中的 bbox 结构可能为嵌套 list 或 numpy array
    bbox = bbox_list[0] if isinstance(bbox_list[0], (list, tuple, np.ndarray)) else bbox_list
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # 异常阻断
    if bbox_width <= 0 or bbox_height <= 0:
        return

    # -----------------------------------------------
    # 等比例 Letterbox 归一化
    # 以长边为统一缩放基准，短边方向居中偏移，保持原始宽高比
    # -----------------------------------------------
    scale    = max(bbox_width, bbox_height)
    x_offset = (scale - bbox_width)  / (2.0 * scale)  # 水平居中偏移
    y_offset = (scale - bbox_height) / (2.0 * scale)  # 垂直居中偏移

    # 初始化一维特征向量 (开头声明尺度)
    flattened_data = [target_size, target_size]
    
    for i in range(len(keypoints)):
        raw_x = keypoints[i][0]
        raw_y = keypoints[i][1]
        
        # 等比例归一化 + 居中偏移（替代原来分轴压缩的方式）
        norm_x = ((raw_x - x_min) / scale + x_offset) * target_size
        norm_y = ((raw_y - y_min) / scale + y_offset) * target_size
        
        # 边界截断，防止特征溢出导致步态网络报错
        norm_x = max(0.0, min(target_size, float(norm_x)))
        norm_y = max(0.0, min(target_size, float(norm_y)))
        
        flattened_data.append(round(norm_x, 6))
        flattened_data.append(round(norm_y, 6))
        # 越界防护：scores 与 keypoints 长度理论相同，但做保险处理
        score = scores[i] if i < len(scores) else 0.0
        flattened_data.append(round(float(score), 8))
        
    # 数据落盘
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    txt_content = ",".join(map(str, flattened_data))
    
    with open(txt_path, 'w', encoding='utf-8') as f_out:
        f_out.write(txt_content)

# ==========================================
# 3. 主流程与树状分叉生成
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MMPose 树状提取、透明 PNG 生成及归一化 TXT 导出工具")
    parser.add_argument('--input', type=str, required=True, help='输入图像所在的根目录')
    parser.add_argument('--output', type=str, required=True, help='结果保存的根目录')
    parser.add_argument('--device',   type=str,   default='cuda:0', help='计算设备 (如 cuda:0, cpu)')
    parser.add_argument('--size',     type=float, default=128.0,    help='步态网络归一化的画布尺寸 (默认 128.0)')
    parser.add_argument('--black_bg', action='store_true',
                        help='启用后输出黑底透明PNG；不传则保留原始背景骨架PNG')
    args = parser.parse_args()

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)

    img_paths = get_image_paths(input_root)
    if not img_paths:
        print(f"致命错误：在 {input_root} 下未找到任何图像文件。")
        return

    print(f"共发现 {len(img_paths)} 张待处理图像。开始初始化模型...")
    inferencer = MMPoseInferencer(pose2d='human', device=args.device)

    # 引入 tqdm 显示进度条
    from tqdm import tqdm
    
    # 使用 tqdm 包裹循环
    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc="处理进度"):
        # --- A. 目录树拓扑映射计算 ---
        # 使用 input_root（已 abspath）而非 args.input，避免相对路径陷阱
        if os.path.isfile(input_root):
            rel_path = os.path.basename(img_path)
        else:
            rel_path = os.path.relpath(img_path, input_root)

        # 逐张推理，确保图像路径与结果严格一一对应
        result = next(iter(inferencer(
            img_path,
            return_vis=True,
            draw_heatmap=False,
            black_background=args.black_bg,   # 由命令行参数控制
            show=False
        )))
            
        # 拆分相对路径
        # rel_dir = 图像相对于 input_root 的子目录层级 (例如: 1.25/侧/rgb)
        # file_core = 纯文件名不含扩展 (例如: track2_00019)
        rel_dir = os.path.dirname(rel_path)
        file_core, _ = os.path.splitext(os.path.basename(rel_path))
        
        # 1. 扩充三大并行父级目录（绑定在对应样本ID目录下）
        # 新结构：output_root / 1.25 / 侧 / rgb / jsons / ...
        sample_out_dir = os.path.join(output_root, rel_dir)
        json_root_dir = os.path.join(sample_out_dir, 'jsons')
        png_root_dir = os.path.join(sample_out_dir, 'pngs')
        txt_root_dir = os.path.join(sample_out_dir, 'txts')
        
        # 2. 拼接文件路径
        out_json_path = os.path.join(json_root_dir, f"{file_core}.json")
        out_png_path = os.path.join(png_root_dir, f"{file_core}.png")
        out_txt_path = os.path.join(txt_root_dir, f"{file_core}.txt")
        
        # 3. 级联创建父级目录（过滤空字符串，防止极端路径场景崩溃）
        for _dir in [json_root_dir, png_root_dir, txt_root_dir]:
            if _dir:
                os.makedirs(_dir, exist_ok=True)

        predictions = result.get('predictions', [])

        # --- B. 结构化 JSON 落盘 ---
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=4, cls=NumpyEncoder)

        # --- C. 步态一维 TXT 特征落盘 (从内存数据直接处理) ---
        # 这里的 args.size 就是原本 j2t.py 里面的 target_size
        extract_and_save_txt(predictions, out_txt_path, args.size)

        # --- D. 透明 PNG 转换与渲染落盘 ---
        vis_imgs = result.get('visualization', [])
        if vis_imgs:
            bgr_matrix = vis_imgs[0]
            if args.black_bg:
                # 黑底模式：去除黑色背景，生成透明 BGRA PNG
                output_matrix = make_transparent_png(bgr_matrix)
            else:
                # 原始背景模式：直接保存带背景的骨架叠加图（BGR，无需转透明）
                if bgr_matrix.dtype != np.uint8:
                    bgr_matrix = (bgr_matrix * 255).clip(0, 255).astype(np.uint8)
                output_matrix = bgr_matrix
            cv2.imwrite(out_png_path, output_matrix)

    print("\n所有任务执行完毕，目录树结构及 txt 特征提取已成功同步。")

if __name__ == '__main__':
    main()