# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/24
from paddlevision.models import resnet50
import paddle
import paddle.fluid as fluid
import numpy as np

import torch


with fluid.dygraph.guard():
    # model = resnet50()
    # in_np = np.random.rand(4, 3, 224, 224).astype("float32")
    # in_var = fluid.dygraph.to_variable(in_np)
    # out = model(in_var)
    # print(out.shape)
    resnet34_pd_path = r"E:\Code\Python\PaddleSeg\paddlevision\models\resnet34"
    resnet50_pd_path = r"E:\Code\Python\PaddleSeg\paddlevision\models\resnet50"
    resnet101_pd_path = r"E:\Code\Python\PaddleSeg\paddlevision\models\resnet101"
    resnet152_pd_path = r"E:\Code\Python\PaddleSeg\paddlevision\models\resnet152"

    model_pd = fluid.dygraph.load_dygraph(resnet152_pd_path)
    print(model_pd[0]['conv1.weight'])

    resnet34_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\resnet34-333f7ec4.pth"
    resnet50_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\resnet50-19c8e357.pth"
    resnet101_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\resnet101-5d3b4d8f.pth"
    resnet152_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\resnet152-b121ed2d.pth"
    model_torch = torch.load(resnet152_torch_path)
    # print(model_torch['conv1.weight'])
