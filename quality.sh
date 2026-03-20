# --input           : 这里填写之前交给 ByteTrack/跑出来的主干跟踪输出目录 (EXP_TRACKED)。
# --output          : 在清除了所有的腿部被遮挡或者只有半截身体的照片后剩下的【绝对安全区】文件夹名 (建议写作 EXP_CLEAN)。
# --min-head-score  : [核心拦截防御] 专门针对相机太低导致的“无头/断头”情况！头部的 5 个探测点中最起码得有一个清楚的（推荐 0.2 以上）。
# --min-leg-score   : 下半身 6 个相连骨骼点的平均看清程度 (阈值定在 0.3 一般比较科学)。
# --min-ankle-score : 重点保护指标 - 这是两只【脚踝】的单独审判线！如果只为了抽躯干可以不管，专门研究步态动作，连脚底都不能出现马赛克，至少要定到 0.1。

CUDA_VISIBLE_DEVICES=1 python quality.py \
    --input ../by_test \
    --output ../output/quality \
    --min-head-score 0.2 \
    --min-leg-score 0.3 \
    --min-ankle-score 0.1 \

