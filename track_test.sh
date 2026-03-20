# --input   : 原图所在的顶级目录路径，脚本会在内寻找所有的 png/jpg 图
# --output  : 树状输出结果目录
# --padding : 额外向外扩张多少像素来裁剪。有些时候 bbox 提供得太紧会吃手吃脚，增加一点 padding 会更好

CUDA_VISIBLE_DEVICES=1 python test.py \
    --input ../fps \
    --output ../output/crop_results \
    --padding 15 
