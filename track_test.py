import os
import cv2
import argparse
import numpy as np
import math
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

class RobustTracker:
    """一个极为健壮的基于中心点距离补偿 (Centroid Distance) 的目标追踪器
    - 废弃了单纯的 IoU 匹配，因为探测框如果突变会导致 IoU 瞬间归零从而换 ID
    - 改为跟踪人物中心点特征，匹配容错率极高
    - 具有记忆效应，人物被遮挡/漏检时可以最多保留 30 帧 (抗丢失)
    """
    def __init__(self, max_disappeared=30, max_dist_ratio=1.5):
        self.tracks = {}         # {track_id: {'center': (cx, cy), 'diag': diagonal_len}}
        self.disappeared = {}    # {track_id: frames}
        self.next_id = 1         
        self.max_disappeared = max_disappeared
        self.max_dist_ratio = max_dist_ratio  # 容许中心点移动的最大倍率 (相对于人物框对角线)

    def update(self, new_bboxes):
        if len(new_bboxes) == 0:
            for tid in list(self.tracks.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            return []

        # 1. 获取当前帧目标的中心点与尺寸特征
        input_feats = []
        for obj in new_bboxes:
            b = obj['bbox']
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            diag = math.hypot(b[2]-b[0], b[3]-b[1])
            input_feats.append({'center': (cx, cy), 'diag': diag, 'bbox': b, 'orig': obj})

        # 2. 空记录则全部分配新 ID
        if len(self.tracks) == 0:
            tracked_objects = []
            for feat in input_feats:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = feat
                self.disappeared[tid] = 0
                obj = feat['orig'].copy()
                obj['track_id'] = tid
                tracked_objects.append(obj)
            return tracked_objects

        track_ids = list(self.tracks.keys())
        track_feats = list(self.tracks.values())

        # 3. 计算相对距离矩阵
        D = np.zeros((len(track_ids), len(input_feats)))
        for i, t_feat in enumerate(track_feats):
            for j, i_feat in enumerate(input_feats):
                dist = math.hypot(t_feat['center'][0] - i_feat['center'][0], 
                                  t_feat['center'][1] - i_feat['center'][1])
                # 标准化距离 = 像素移动距离 / 旧框对角线长度
                # 这使得模型能无缝适应超远景和贴脸大特写的追踪，不受绝对像素约束
                norm_dist = dist / (t_feat['diag'] + 1e-5)
                D[i, j] = norm_dist

        matched_track_indices = set()
        matched_new_indices = set()
        tracked_objects = []

        # 4. 贪心寻找最佳对应，解决拥挤遮挡时的交集问题
        while True:
            if D.size == 0 or np.min(D) > self.max_dist_ratio:
                break
            min_idx = np.unravel_index(np.argmin(D), D.shape)
            t_idx, n_idx = min_idx

            tid = track_ids[t_idx]
            self.tracks[tid] = input_feats[n_idx]
            self.disappeared[tid] = 0
            
            obj = input_feats[n_idx]['orig'].copy()
            obj['track_id'] = tid
            tracked_objects.append(obj)

            D[t_idx, :] = np.inf
            D[:, n_idx] = np.inf
            
            matched_track_indices.add(t_idx)
            matched_new_indices.add(n_idx)

        # 5. 清理没被匹配的旧目标 (进入遮挡或离开画面)
        for i, tid in enumerate(track_ids):
            if i not in matched_track_indices:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

        # 6. 新进画面的边缘目标注册为全新 ID
        for j, feat in enumerate(input_feats):
            if j not in matched_new_indices:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = feat
                self.disappeared[tid] = 0
                
                obj = feat['orig'].copy()
                obj['track_id'] = tid
                tracked_objects.append(obj)

        return tracked_objects


def get_image_paths(input_dir: str) -> list:
    if os.path.isfile(input_dir):
        return [input_dir]
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description="多目标人物检测裁剪与健壮追踪脚本")
    parser.add_argument('--input', type=str, required=True, help='输入图像所在的根目录')
    parser.add_argument('--output', type=str, required=True, help='裁剪图像保存的根目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备')
    parser.add_argument('--padding', type=int, default=15, help='裁剪像素膨胀值')
    parser.add_argument('--min-score', type=float, default=0.5, help='目标最低置信度阈值（过滤背景杂像）')
    args = parser.parse_args()

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    img_paths = get_image_paths(input_root)
    
    if not img_paths:
        print(f"在 {input_root} 未找到图像文件。")
        return

    inferencer = MMPoseInferencer(pose2d='human', device=args.device)
    
    # 全局追踪变量
    tracker = None
    current_dir = None

    for idx, img_path in tqdm(enumerate(img_paths), total=len(img_paths), desc="序列裁剪跟综进度"):
        
        # 1. 结构与拓扑
        if os.path.isfile(input_root):
            rel_path = os.path.basename(img_path)
            rel_dir = ""
        else:
            rel_path = os.path.relpath(img_path, input_root)
            rel_dir = os.path.dirname(rel_path)
            
        filename = os.path.basename(rel_path)
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        # 换了一个视频文件夹，立刻清空并重置 Tracker 的跟踪视野
        if rel_dir != current_dir:
            tracker = RobustTracker(max_disappeared=30, max_dist_ratio=1.5)
            current_dir = rel_dir

        # 2. 执行推理
        result = next(iter(inferencer(img_path, return_vis=False, show=False)))
        predictions = result.get('predictions', [])
        
        if len(predictions) > 0:
            preds_list = predictions[0] if isinstance(predictions[0], list) else predictions
            
            # 清点本帧视野内所有的高置信度方框
            valid_bboxes = []
            for person in preds_list:
                bbox_list = person.get('bbox', [])
                score = person.get('bbox_score', 0)
                if len(bbox_list) > 0 and score >= args.min_score:
                    b_rect = bbox_list[0] if isinstance(bbox_list[0], (list, tuple, np.ndarray)) else bbox_list
                    valid_bboxes.append({'bbox': b_rect, 'score': score})
            
            # 3. 让 Robust Tracker 给他们套上身份 ID
            tracked_objects = tracker.update(valid_bboxes)
            
            if len(tracked_objects) > 0:
                orig_img = cv2.imread(img_path)
                if orig_img is not None:
                    h, w = orig_img.shape[:2]
                    
                    # 取出原图的主名与扩展名，例如 'c0000048486_f0' 和 '.png'
                    name_core, ext = os.path.splitext(filename)
                    # 利用 '_' 分割并获取最后的顺序编码部分（如 'f0'），如果没有下划线则回退使用原名
                    fps_order = name_core.split('_')[-1] if '_' in name_core else name_core

                    for obj in tracked_objects:
                        tid = obj['track_id']
                        x_min, y_min, x_max, y_max = obj['bbox'][:4]
                        
                        pad = args.padding
                        x1 = max(0, int(x_min) - pad)
                        y1 = max(0, int(y_min) - pad)
                        x2 = min(w, int(x_max) + pad)
                        y2 = min(h, int(y_max) + pad)
                        
                        if x2 > x1 and y2 > y1:
                            crop_img = orig_img[y1:y2, x1:x2]
                            
                            # ID 在前，顺序在后的形式：如 id1_f0.png, id2_f0.png
                            final_name = f"id{tid}_{fps_order}{ext}"
                            out_path = os.path.join(out_dir, final_name)
                            
                            cv2.imwrite(out_path, crop_img)

if __name__ == '__main__':
    main()
