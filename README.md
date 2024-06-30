# HeightAnyone

HeightAnyone is a framework built on top of the state-of-the-art depth estimation model, [DepthAnyThingV2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2), and [YOLOv9](https://github.com/WongKinYiu/yolov9) for segmentation, designed for human height estimation, This framework <ins> **does not require   any** </ins> information about the camera or scene parameters, except for at least one vertical known height in the video. 


![demo](https://github.com/MohammadDallash/Height-Anything/assets/105324962/dc709f2d-07cf-44f1-b975-718c0242a632)


## Usage

**1. Install Requirements:**

```
pip install -r requirements.txt
```
**2. Download Models (around 1.3GB)**
```
./download_models.sh
```

**3. Follow along the steps in the pipeline notebook**



## Minimum Requirements

- GPU with at least 3GB of VRAM

