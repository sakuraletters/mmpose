环境安装 mamba env create -f environment_mmpose.yml  
 
mim install "mmpose>=1.1.0"
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