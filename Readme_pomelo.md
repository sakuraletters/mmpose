环境安装见 environment_mmpose.yml

验证安装使用mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest . 
    python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap

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