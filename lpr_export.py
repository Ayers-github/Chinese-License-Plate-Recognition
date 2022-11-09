import cv2
import numpy as np
import torch

from models.LPRNet import LPRNet, CHARS
from test_plate_color import lpr, transform

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
print(len(CHARS))
# 注意，调整lpr在前面加载，不然会后面的SimplePredict影响，SimplePredict中设置了env为-1
# 加载车牌识别模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
modelc.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=device))
modelc.to(device).eval()



"""lpr"""
# data_input = torch.rand(2, 3, 24, 94).to(device)
# pl = lpr(data)
# print(pl)

"""替换lpr"""
data = cv2.imread("8.jpg")
cutout_in_lpr = cv2.resize(data, (94, 24))  # BGR ，resize成94*24
# cv2.imshow("processed_img", cutout_in_lpr)
# cv2.waitKey(0)
cutout_in_lpr_re = transform(cutout_in_lpr)  # 对输入进行处理
data_input = torch.tensor([cutout_in_lpr_re]).to(device)
print(data_input[0][0][0])
preds = modelc(data_input)  # 车牌识别
# print(preds[0][0])
# print(preds[0][-1])
print(preds)
print(preds.shape)
prebs = preds.cpu().detach().numpy()
preb_labels = list()
for w in range(prebs.shape[0]):
    preb = prebs[w, :, :]  # 获取第w个车牌的识别结果
    print(preb.shape)
    preb_label = list()  # 创建第w个车牌的识别结果的label序数存储咧白哦
    for j in range(preb.shape[1]):  # 遍历车牌预选的18个位置向量
        preb_label.append(np.argmax(preb[:, j], axis=0))  # 每个车牌位置向量中，都有68个位置的概率，获取所有68个位置中最大概率的序数
    print(preb_label)  # 打印18个位置最有可能的车牌数序号，下面会使用此原始序号进行去重操作
    no_repeat_blank_label = list()
    pre_c = preb_label[0]
    print("pre_c", pre_c)

    # 如果第一位不是终止符，即"-",那么将其添加到no_repeat_blank_label列表
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)

    # 遍历18位车牌位置向量
    for c in preb_label:  # dropout repeate label and blank label
        if (c == pre_c) or (c == len(CHARS) - 1):  # 如果这个位置的向量等于终止符序号或者是等于pre_c
            if c == len(CHARS) - 1:  # 如果等于空白符，那么将pre_c的值设置为空白符
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    print(no_repeat_blank_label)
    preb_labels.append(no_repeat_blank_label)

plat_num_in_lpr = np.array(preb_labels)
print(plat_num_in_lpr)

for j in range(plat_num_in_lpr.shape[0]):
    lb = ""
    for i in plat_num_in_lpr[j]:
        lb += CHARS[i]
    print(lb)
plate_num = '%s' % lb
print(plate_num)
with torch.no_grad():
    jit_model = torch.jit.trace(modelc, data_input)
torch.onnx._export(modelc, data_input, "lpr2d.onnx", export_params=True, opset_version=11)
# torch.onnx.export(modelc, data_input, "lpr2d.onnx", verbose=False, opset_version=12)
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "lpr2d.onnx"
model_quant = "lpr2d_quant.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)