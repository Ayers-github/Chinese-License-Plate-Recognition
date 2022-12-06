# Chinese-License-Plate-Recognition
中国多车牌检测、车牌角度矫正、车牌识别、危险品车辆识别安卓版本，使用ncnn进行推理。

车牌检测模型：yolov5face

车牌矫正：透视变换
车牌识别：crnn

车牌颜色识别：单层卷积神经网络

步骤：

配置sdk和ndk编译后即可运行
minSdkVersion:24
ndk:24.0.8215888
cmake:3.10.2

tips：需要在yolov5ncnn_jnitest.cpp文件中修改包路径为自己的包路径，一共有四处（可以通过搜索tencent找到这四处)

如果需要在其他项目中使用车牌识别模块，直接调用PlateRecognition.java类即可。
