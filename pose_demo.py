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

def get_box_area(person_data: dict) -> float:
    """计算单个边界框的几何面积，用于在极端视角中准确筛选出占幅最大的主角人物"""
    bbox_list = person_data.get('bbox', [])
    if not bbox_list:
        return 0.0
    bbox = bbox_list[0] if isinstance(bbox_list[0], (list, tuple, np.ndarray)) else bbox_list
    if len(bbox) < 4:
        return 0.0
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


# COCO 17关键点标准连线对（左侧橙色，右侧绿色，中轴蓝色）
COCO_SKELETON = [
    (0,  1,  (255, 128,   0)),  # 鼻 - 左眼
    (0,  2,  (255, 128,   0)),  # 鼻 - 右眼
    (1,  3,  (255, 128,   0)),  # 左眼 - 左耳
    (2,  4,  (255, 128,   0)),  # 右眼 - 右耳
    (3,  5,  (255, 128,   0)),  # 左耳 - 左肩（头身连接）
    (4,  6,  (255, 128,   0)),  # 右耳 - 右肩（头身连接）
    (5,  6,  (  0, 128, 255)),  # 左肩 - 右肩
    (5,  7,  (  0, 255,   0)),  # 左肩 - 左肘
    (7,  9,  (  0, 255,   0)),  # 左肘 - 左手
    (6,  8,  (255, 128,   0)),  # 右肩 - 右肘
    (8, 10,  (255, 128,   0)),  # 右肘 - 右手
    (5, 11,  (  0, 128, 255)),  # 左肩 - 左髋
    (6, 12,  (  0, 128, 255)),  # 右肩 - 右髋
    (11,12,  (  0, 128, 255)),  # 左髋 - 右髋
    (11,13,  (  0, 255,   0)),  # 左髋 - 左膝
    (13,15,  (  0, 255,   0)),  # 左膝 - 左踝
    (12,14,  (255, 128,   0)),  # 右髋 - 右膝
    (14,16,  (255, 128,   0)),  # 右膝 - 右踝
]


def draw_skeleton_on_image(img_bgr: np.ndarray, person_data: dict, black_bg: bool) -> np.ndarray:
    """在图像上手工绘制单人骨架，完全不依赖 MMPose 渲染器。"""
    if black_bg:
        canvas = np.zeros_like(img_bgr)
    else:
        canvas = img_bgr.copy()

    kpts   = person_data.get('keypoints', [])
    scores = person_data.get('keypoint_scores', [])

    # 绘制骨骼连线
    for (i, j, color) in COCO_SKELETON:
        if i >= len(kpts) or j >= len(kpts):
            continue
        si = scores[i] if i < len(scores) else 0.0
        sj = scores[j] if j < len(scores) else 0.0
        if si < 0.1 or sj < 0.1:
            continue
        xi, yi = int(kpts[i][0]), int(kpts[i][1])
        xj, yj = int(kpts[j][0]), int(kpts[j][1])
        cv2.line(canvas, (xi, yi), (xj, yj), color, 2, cv2.LINE_AA)

    # 绘制关键点圆点
    kpt_colors = [
        (255,128,0),(255,128,0),(255,128,0),(255,128,0),(255,128,0),  # 0-4
        (0,255,0),(255,128,0),(0,255,0),(255,128,0),(0,255,0),(255,128,0),  # 5-10
        (0,255,0),(255,128,0),(0,255,0),(255,128,0),(0,255,0),(255,128,0),  # 11-16
    ]
    for i, (x, y) in enumerate(kpts):
        s = scores[i] if i < len(scores) else 0.0
        if s < 0.1:
            continue
        color = kpt_colors[i] if i < len(kpt_colors) else (0, 255, 255)
        cv2.circle(canvas, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

    return canvas

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

    # 3. 提取画面面积最大的主目标避开背景路人干扰 (此时 predictions 已经是字典列表 [{...}, {...}])
    best_instance = max(predictions, key=get_box_area)
    
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

        # --- [精准检测框定轨法：强制单人提取] ---
        # 1. 第一次快速推理：不开可视化，单纯为了获取所有人（包括干扰路人）的检测框
        fast_result = next(iter(inferencer(img_path, return_vis=False, show=False)))
        all_preds = fast_result.get('predictions', [])
        
        # MMPose 结果解包降维
        if all_preds and isinstance(all_preds[0], list):
            all_preds = all_preds[0]
            
        if all_preds:
            # 2. 筛选出主目标单人：计算包围框面积，挑选在画面中绝对物理面积最大的主角
            best_person = max(all_preds, key=get_box_area)

            # 3. predictions 数据已是单人，直接用 first-pass 结果
            result = fast_result
            result['predictions'] = [[best_person]]
        else:
            result = fast_result
            
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

        # --- D. 手工绘制单人骨架 PNG（完全绕过 MMPose 内置渲染器，避免多人残影）---
        predictions = result.get('predictions', [])
        flat_preds = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
        if flat_preds:
            img_bgr = cv2.imread(img_path)
            skeleton_img = draw_skeleton_on_image(img_bgr, flat_preds[0], args.black_bg)
            cv2.imwrite(out_png_path, skeleton_img)

    print("\n所有任务执行完毕，目录树结构及 txt 特征提取已成功同步。")

if __name__ == '__main__':
    main()