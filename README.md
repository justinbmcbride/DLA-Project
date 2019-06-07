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
* In surveying the web, this **[post](https://jkjung-avt.github.io/yolov2/)** was found. It basically follows the instructions found **[here](https://pjreddie.com/darknet/yolov2/)** and describes a few modifications to get it to work on the TX2.
* The repository with the YOLOv2 pretrained weights were downloaded from **[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet).**
* The first few lines of `Makefile` was updated to reflect TX2 hardware configuration.
