# 反斜杠 (\) 必须作为该行的最后一个字符，其后绝对不能包含任何空格或注释
#--black_bg参数控制是否需要可视化人物背景，需要背景可注释掉
CUDA_VISIBLE_DEVICES=2 python pose_demo.py \
    --input ../1.25 \
    --output ../output/keypoint_results \
    --black_bg