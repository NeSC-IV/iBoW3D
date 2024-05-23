# iBoW3D
Code for paper iBoW3D: Place Recognition Based on Incremental and General Bag of Words in 3D Scans (ICRA 2024).

![overview](https://github.com/NeSC-IV/iBoW3D/blob/main/fig/pipeline.png "overview")
<p align="center">Fig.1 The method overview</p>

## Video
- [Bilibili](https://www.bilibili.com/video/BV1bC4y177Tr/?spm_id_from=333.999.0.0&vd_source=7809b69c2f87086cb4eb0391049451c1)
- [Youtube](https://www.youtube.com/watch?v=K5w-44xg4VI&t=1s)

## Prerequisites
We ran the code in Ubuntu 18.04 and we also used libraries listed below.
- [Open3D](https://www.open3d.org/)
- [OpenCV](https://github.com/opencv/opencv)
- [Eigen 3](https://eigen.tuxfamily.org/dox/)


## Features
We use [D3Feat](https://github.com/XuyangBai/D3Feat?tab=readme-ov-file) to obtain keypoints and local features in advance.

## Compile and Run
Before compile the package, some related paths should be added in the **main.cpp** file.

```
git clone https://github.com/NeSC-IV/iBoW3D.git
mkdir build
cd build
cmake ..
make
./ibow3d
```

## Acknowledgement
Thanks to the code of [D3Feat](https://github.com/XuyangBai/D3Feat?tab=readme-ov-file).
