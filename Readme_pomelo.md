#  PyTorch 版本需要调整(看CUDA版本 )

# 环境安装 mamba env create -f environment_mmpose.yml （里面有cuda+torch，可以手动替换）

#  安装 MMLab 的包管理器 MIM (用于下载预编译的超级 C++ 算子)
pip install -U openmim
mim install mmengine

# 必须用 mim 安装带着完整 C++ 算子的 mmcv，绝对不能直接 pip install mmcv！
mim install "mmcv==2.1.0"

# 最后，通过官方渠道安装最新版的目标检测引擎 mmdetection
mim install "mmdet>=3.0.0" 

# 碰到诸如 `ModuleNotFoundError: No module named 'mmcv._ext'` 的报错，一定是第 4 步没有走 MIM 编译。只需 `pip uninstall mmcv -y` 然后重新跑第 4 步即可。

验证安装使用mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest . 
    python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
质量筛选使用 sh quality.sh
track_test是尝试使用该模型进行人物提取，但实测效果证明该模型不适合此工作，文件保留，但不使用该模型提取

pose提取(json\txt\png三模态)。 sh pose.sh 内置路径修改

sh文件中--black_bg参数控制是否需要可视化人物背景，需要背景可注释掉

文件保存结构为
../output/keypoint_results/
└──日期/
    └──ID/
        └──视角/  
            ├── jsons/
            │   └── track.json
            ├── pngs/
            │   └── track.png
            └── txts/
                └── track.txt