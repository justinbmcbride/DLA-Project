# DLA-Project
#### Objective
This project is part of the interview process for DLA at GD-MS. It is required to be done on the NVIDIA Jetson TX2 developer kit and shipped back to DLA within two weeks of receipt (ship back by 6/21/19). The objective of my project is to run a small YOLO (you only look once) model on the kit and identify objects through the camera.

#### Procedure
* Survey literature and the web for code already written and models already for small scale YOLO
* Clone repositories
* Install necessary software
* Configure hardware and software
* Test configuration on how accurate it can identify common objects

#### Results
* In surveying the web, instructions for running the TX2 using yolov2 (you only look once) darknet was found **[here](https://jkjung-avt.github.io/yolov2/)**.
* The repository with the yolov2 pretrained weights were downloaded from **[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet).**
* The first few lines of `Makefile` was updated to reflect TX2 hardware configuration.
* In the demo repo, a webcam was used, but it is requred by DLA to only use hardware available on the Jetson, so more digging was required.
* As the darknet repo was explored, it was noticed that there was a yolov3 was available. The instructions for that setup is **[https://jkjung-avt.github.io/yolov3/](https://jkjung-avt.github.io/yolov3/)** 
* Opencv 3.3.1 came with Jetpack, but version 3.4.* is required for gstreamer functionality. Install instructions are here **[https://jkjung-avt.github.io/opencv-on-nano/](https://jkjung-avt.github.io/opencv-on-nano/)**
* The `Makefile` script was modified from
```
GPU=1
CUDNN=1
OPENCV=1
......
ARCH= -gencode arch=compute_53,code=[sm_53,compute_53] \
-gencode arch=compute_62,code=[sm_62,compute_62]
```
* At first, full model did not work as it caused the system to crash. Then used this command to execute demo `$./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3.weights traffic.mkv`
* However, no objects were identified. It may be that the *mkv format is not recognizable.
* Download traffic video in mp4 format using `youtube-dl -f 18 https://www.youtube.com/watch?v=wqctLW0Hb_0&feature=youtu.be
`

