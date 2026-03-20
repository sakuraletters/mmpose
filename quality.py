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
    parser = argparse.ArgumentParser(description="基于 MMPose 骨架透视评分的残缺遮挡过滤系统")
    parser.add_argument('--input', type=str, required=True, help='ByteTrack等初步追踪好的图片根目录')
    parser.add_argument('--output', type=str, required=True, help='质量清洗合格且原样保留了树形结构的存放目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备 (如 cuda:0)')
    # 以下核心参数可自由用各种数值做步态“严格程度”的微调
    parser.add_argument('--min-head-score', type=float, default=0.2, help='头部5个关键点必定有1个清晰及格(专杀断头帧)')
    parser.add_argument('--min-leg-score', type=float, default=0.3, help='下肢(跨膝踝)6个关键点平均可见度合格线')
    parser.add_argument('--min-ankle-score', type=float, default=0.1, help='双脚踝最低要求得分(防小腿完全被挡/出框)')
    
    args = parser.parse_args()

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    
    img_paths = get_image_paths(input_root)
    if not img_paths:
        print(f"在 {input_root} 未找到任何图像文件，请检查目录是否挂载正确。")
        return

    print(f"已排队 {len(img_paths)} 张带有骨干识别框图谱，准备进行像素级结构透视质检...")
    print("模型加载初始化中...")
    
    # 调用自带人体检测与骨架判定的 MMPose 预测全生命流水线组件
    inferencer = MMPoseInferencer(pose2d='human', device=args.device)
    
    kept = 0
    dropped = 0

    import cv2
    for img_path in tqdm(img_paths, desc="执行质量筛选流水线"):
        
        # 0. 强力防御屏障：先用 cv2 探路。
        # ByteTrack 切图时极偶尔会因为坐标溢出或异常切出 0 byte 的损坏/空空如也图片。
        # 如果直接把坏图路径丢给底层 mmcv，它底层 C++ 读取引擎会直接崩溃报错 `!buf.empty()`
        try:
            img_mat = cv2.imread(img_path)
            if img_mat is None or img_mat.size == 0 or img_mat.shape[0] == 0 or img_mat.shape[1] == 0:
                dropped += 1
                continue
            
            # 1. 将物理校验绝对安全的 opencv 内存图直接喂给 MMPose
            result = next(iter(inferencer(img_mat, return_vis=False, show=False)))
        except Exception as e:
            # 任何底层推理失败（如完全找不到人），直接按废片处理
            dropped += 1
            continue
            
        predictions = result.get('predictions', [])
        
        valid = False
        if len(predictions) > 0:
            preds_list = predictions[0] if isinstance(predictions[0], list) else predictions
            
            if len(preds_list) > 0:
                best_instance = max(preds_list, key=lambda x: x.get('bbox_score', 0))
                kpts_score = best_instance.get('keypoint_scores', [])
                
                # 2. 从逻辑与解剖学角度打分 (COCO 标准格式说明：)
                # 0~4 为无关紧要头部；5~10 为胳膊。
                # 11~12：胯部 (Hip)
                # 13~14：膝盖 (Knee)
                # 15~16：脚踝 (Ankle)
                
                # 在步态识别当中，下半身是特征核心，但上半身(尤其头部)决定了高度对齐基准。
                # 如果没有头只有腿，步态轮廓提取会完全失败，因此上下半身都必须接受安检！
                if len(kpts_score) >= 17:
                    # 0:鼻子, 1,2:眼睛, 3,4:耳朵。这5个点构成了脑袋的全部特征。
                    head_scores = kpts_score[0:5]
                    legs_scores = kpts_score[11:17]
                    
                    # 判决法则 A：【地狱级防断头查杀】
                    # 不能看全体上半身平均值，因为如果此时他的手臂(比如图里穿着黑羽绒服的手垂下来)非常清晰，
                    # 就会把平均分拉高，从而掩盖了“头已经完全出框”的致命事实！
                    # 核心逻辑：脑袋上的5个关键点，哪怕看背影也得能识别到至少一只耳朵。
                    # 如果这5个点全军覆没(最高分都极低)，说明头被物理切掉了，当场判死！
                    max_head = max(head_scores)
                    
                    # 判决法则 B：总体腿部置信度均值不能低到马赛克状态
                    avg_leg = sum(legs_scores) / len(legs_scores)
                    
                    # 判决法则 C：双脚踝绝对不能脱离镜头边界致死
                    left_ankle = kpts_score[15]
                    right_ankle = kpts_score[16]
                    
                    if max_head >= args.min_head_score and avg_leg >= args.min_leg_score and min(left_ankle, right_ankle) >= args.min_ankle_score:
                        valid = True

        # 3. 行刑 / 放行决议：所有达标者继续按照相同的路径复制到下个流程，落选者一并遗弃
        if valid:
            rel_path = os.path.relpath(img_path, input_root)
            out_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # 无损且极速通过底层 IO 进行数据镜像
            shutil.copy2(img_path, out_path)
            kept += 1
        else:
            dropped += 1
            
    print("\n\n======== 步态数据结构病理清洗报告 ========")
    print(f"[通过率]: {(kept/(kept+dropped+1e-5))*100:.1f}%")
    print(f"[√] 成功打捞安全帧: {kept} 张")
    print(f"[X] 处死下肢缺损帧: {dropped} 张")
    print("===========================================")

if __name__ == '__main__':
    main()
