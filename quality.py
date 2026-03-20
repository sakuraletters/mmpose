import argparse
import os
import shutil
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer


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
    parser = argparse.ArgumentParser(description="基于 MMPose 全身完整度检测的步态质量筛选系统")
    parser.add_argument('--input',  type=str, required=True, help='ByteTrack 追踪后的输出目录')
    parser.add_argument('--output', type=str, required=True, help='清洗后合格序列的保存目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备 (如 cuda:0)')
    # 核心参数：17个关键点中至少有多少个坐标落在图片边界内
    # 完整全身可见: ~15-17   |   大半身可见: ~11-13   |   半身或更差: ~6-9
    # 严格推荐 13，极严格可升至 14
    parser.add_argument('--min-kpts-visible', type=int, default=13,
                        help='17个关键点中至少有多少个坐标在图片边界内 (越高越严格, 推荐13)')

    args = parser.parse_args()
    input_root  = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)

    img_paths = get_image_paths(input_root)
    if not img_paths:
        print(f"在 {input_root} 未找到任何图像文件，请检查目录是否挂载正确。")
        return

    print(f"已排队 {len(img_paths)} 张图像，完整度阈值: {args.min_kpts_visible}/17 关键点在画内...")
    inferencer = MMPoseInferencer(pose2d='human', device=args.device)

    kept = 0
    dropped = 0
    from collections import defaultdict
    import cv2
    # key: 相对子文件夹路径, value: [kept_count, total_count]
    folder_stats = defaultdict(lambda: [0, 0])

    for img_path in tqdm(img_paths, desc="执行全身完整度质检"):
        try:
            img_mat = cv2.imread(img_path)
            if img_mat is None or img_mat.size == 0:
                dropped += 1
                continue
            result = next(iter(inferencer(img_mat, return_vis=False, show=False)))
        except Exception:
            dropped += 1
            continue

        predictions = result.get('predictions', [])
        valid = False
        if predictions:
            preds_list = predictions[0] if isinstance(predictions[0], list) else predictions
            if preds_list:
                best = max(preds_list, key=lambda x: x.get('bbox_score', 0))
                keypoints  = best.get('keypoints', [])
                kpts_score = best.get('keypoint_scores', [])
                img_h, img_w = img_mat.shape[:2]

                # ============================================================
                # 【核心逻辑：双层可见度检测，专门对抗幻觉分数】
                # 问题根源：MMPose 对裁剪图做骨架推算时，会把头肩等不可见部位
                # 的幻觉坐标投影到图片内部，并给出"中等幻觉分数"(0.3~0.5)。
                # 纯粹用坐标边界 + 低阈值(0.1) 无法区分"真实可见"与"幻觉在画内"。
                #
                # 解决方案：
                # - 宽松层(score>0.1 + 在画内): 用于基础坐标合法性过滤
                # - 严格层(score>0.5 + 在画内): 真实可见关键点才能达到的高置信度
                # 幻觉点: score≈0.3~0.5 → 宽松层通过，严格层被淘汰
                # 真实点: score≈0.6~0.95 → 两层均通过
                # ============================================================

                def is_strict_visible(idx):
                    """关键点真实存在于画内（非幻觉）"""
                    if idx >= len(keypoints) or idx >= len(kpts_score):
                        return False
                    x, y = float(keypoints[idx][0]), float(keypoints[idx][1])
                    return (0 <= x < img_w) and (0 <= y < img_h) and kpts_score[idx] > 0.5

                # 严格可见关键点总数（幻觉点被淘汰）
                strict_visible = sum(1 for i in range(min(17, len(keypoints))) if is_strict_visible(i))

                # 强制要求头部至少 1 点严格可见（鼻/眼/耳，idx 0-4）
                head_strict = any(is_strict_visible(i) for i in range(5))

                # 强制要求双脚踝都严格可见（idx 15, 16）
                ankles_strict = is_strict_visible(15) and is_strict_visible(16)

                valid = (strict_visible >= args.min_kpts_visible and head_strict and ankles_strict)

        folder_key = os.path.relpath(os.path.dirname(img_path), input_root)
        folder_stats[folder_key][1] += 1  # total
        if valid:
            rel_path = os.path.relpath(img_path, input_root)
            out_path  = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(img_path, out_path)
            kept += 1
            folder_stats[folder_key][0] += 1  # kept
        else:
            dropped += 1

    print(f"\n======== 步态质检报告 ========")
    print(f"[通过率] {(kept/(kept+dropped+1e-5))*100:.1f}%")
    print(f"[√] 合格帧: {kept} 张")
    print(f"[X] 不完整帧: {dropped} 张")
    print(f"================================\n")
    print(f"各文件夹过滤详情 (按过滤率从高到低排序):")
    print(f"{'文件夹':<40} {'保留':>6} {'总数':>6} {'过滤率':>8}")
    print("-" * 65)
    sorted_folders = sorted(folder_stats.items(), key=lambda x: (x[1][1]-x[1][0])/max(x[1][1],1), reverse=True)
    for folder, (fkept, ftotal) in sorted_folders:
        drop_rate = (ftotal - fkept) / max(ftotal, 1) * 100
        # 高过滤率的文件夹加星号标注，方便快速发现问题序列
        flag = " <<<" if drop_rate > 60 else ""
        print(f"{folder:<40} {fkept:>6} {ftotal:>6} {drop_rate:>7.1f}%{flag}")
    print("-" * 65)


if __name__ == '__main__':
    main()
