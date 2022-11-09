# Chinese-License-Plate-Recognition
中国多车牌检测、车牌角度矫正、车牌识别、危险品车辆识别

## 此分支为实现车牌识别的ncnn c++版本
# 步骤
1、将lprnet的pytorch模型中的3D卷积算子替换为2D，可以直接将文件LPRNet.py替换为此分支的文件；

2、将pytorch模型导出为onnx模型，可以参考lpr_export.py：

```python
with torch.no_grad():
    jit_model = torch.jit.trace(modelc, data_input)
torch.onnx._export(modelc, data_input, "lpr2d.onnx", export_params=True, opset_version=11)
