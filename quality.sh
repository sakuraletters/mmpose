# MMPose 存在"骨架先验幻觉"：即使头肩出框，模型也会推算出画外坐标并给出中等分数。
# 本脚本统计 17 个关键点中有多少个坐标真实落在图片边界内，以此衡量身体完整程度。
# 完整全身: ~15-17 个在画内 | 半身: ~6-9 个在画内
#
# --input           : ByteTrack 追踪后的输出目录 
# --output          : 清洗后合格序列的存放目录 
# --min-kpts-visible: 17个关键点至少有多少个坐标在画内 (推荐13，越高越严格)

CUDA_VISIBLE_DEVICES=1 python quality.py \
    --input ../by_test \
    --output ../output/quality \
    --min-kpts-visible 14 \

