# Chinese-License-Plate-Recognition
中国多车牌检测、车牌角度矫正、车牌识别、危险品车辆识别

## 此分支为实现车牌识别的ncnn c++版本，20221109暂时上线了LPRnet的ncnn c++版本（仅支持对车牌区域的的车牌识别）。
# 步骤
1、将lprnet的pytorch模型中的3D卷积算子替换为2D，可以直接将文件LPRNet.py替换为此分支的文件；

2、将pytorch模型导出为onnx模型，可以参考lpr_export.py：

```python
with torch.no_grad():
    jit_model = torch.jit.trace(modelc, data_input)
torch.onnx._export(modelc, data_input, "lpr2d.onnx", export_params=True, opset_version=11)
```

3、将onnx文件简化：

```python
onnxsim lpr2d.onnx lpr2d-sim.onnx
```

4、将简化后的onnx模型转为ncnn模型
```python
onnx2ncnn.exe lpr2d-sim.onnx lpr2d.param lpr2d.bin
```

5、将lpr2d.param拖至 https://netron.app/ 页面，查看网络的最后一层的output的name，如我这是131
![image](https://user-images.githubusercontent.com/57164239/200736606-6eec929f-aaea-4e4e-8f34-a4ec5bc40d2a.png)
输入层同理。此处为input.1。

6、使用C++设置依赖项opencv和ncnn，修改ncnn_demo中的第92行
```C++
ex.extract("131", feat)
```
中的131为上一步中的输出name即可。运行ncnn_demo，可以成功识别车牌，注意仅仅能识别车牌区域。

有问题可以提issues 或者加qq群:871797331 询问。群主是https://github.com/we0091234/Chinese_license_plate_detection_recognition项目所有者。
