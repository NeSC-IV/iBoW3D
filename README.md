# iBoW3D
Code for paper iBoW3D: Place Recognition Based on Incremental and General Bag of Words in 3D Scans (ICRA 2024)

![overview](https://github.com/NeSC-IV/iBoW3D/blob/main/fig/pipeline.png "overview")
<p align="center">Fig.1 The method overview</p>

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
