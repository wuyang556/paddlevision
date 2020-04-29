
import math
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# import torch
# import torch.utils.model_zoo as model_zoo
import paddle
import paddle.fluid as fluid
# from ..utils import ReLU
from PaddleVision.models.utils import ReLU

__all__ = ['res2next50']
model_urls = {
    'res2next50': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2next50_4s-6ef7e7bf.pth',
}


class Bottle2neckX(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = fluid.dygraph.Conv2D(inplanes, D*C*scale, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.bn1 = fluid.dygraph.BatchNorm(D*C*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = fluid.dygraph.Pool2D(pool_type="avg", pool_size=3, pool_stride=stride, pool_padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(fluid.dygraph.Conv2D(D*C, D*C, filter_size=3, stride=stride, padding=1, groups=C, bias_attr=False))
          bns.append(fluid.dygraph.BatchNorm(D*C))
        self.convs = fluid.dygraph.container.LayerList(convs)
        self.bns = fluid.dygraph.container.LayerList(bns)

        self.conv3 = fluid.dygraph.Conv2D(D*C*scale, planes * 4, filter_size=1, stride=1, padding=0, bias_attr=False)
        self.bn3 = fluid.dygraph.BatchNorm(planes * 4)
        self.relu = ReLU()

        self.downsample = downsample
        self.width = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # spx = torch.split(out, self.width, 1)
        spx = fluid.layers.split(out, num_or_sections=out.shape[1]//self.width, dim=1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = fluid.layers.concat([out, sp], 1)
        if self.scale != 1 and self.stype=='normal':
          out = fluid.layers.concat([out, spx[self.nums]], 1)
        elif self.scale != 1 and self.stype=='stage':
          out = fluid.layers.concat([out, self.pool(spx[self.nums])],1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2NeXt(fluid.dygraph.Layer):
    def __init__(self, block, baseWidth, cardinality, layers, num_classes, scale=4):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
            scale: scale in res2net
        """
        super(Res2NeXt, self).__init__()

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64
        self.scale = scale

        self.conv1 = fluid.dygraph.Conv2D(3, 64, 7, 2, 3, bias_attr=False)
        self.bn1 = fluid.dygraph.BatchNorm(64)
        self.relu = ReLU()
        self.maxpool1 = fluid.dygraph.Pool2D(pool_type="max", pool_size=3, pool_stride=2, pool_padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = fluid.dygraph.Pool2D(global_pooling=True, pool_type="avg", pool_size=1)
        self.fc = fluid.dygraph.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fluid.dygraph.container.Sequential(
                fluid.dygraph.Conv2D(self.inplanes, planes * block.expansion,
                                     filter_size=1, stride=stride, bias_attr=False),
                fluid.dygraph.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, scale=self.scale, stype='stage'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, scale=self.scale))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = fluid.layers.reshape(x, shape=[x.shape[0], -1])
        x = self.fc(x)

        return x


def res2next50(pretrained=False, **kwargs):
    """    Construct Res2NeXt-50.
    The default scale is 4.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2NeXt(Bottle2neckX, layers=[3, 4, 6, 3], baseWidth=4, cardinality=8, scale = 4, num_classes=1000)
    if pretrained:
        model_path = ""
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
        # model.load_state_dict(model_zoo.load_url(model_urls['res2next50']))
    return model


if __name__ == '__main__':
    # images = torch.rand(1, 3, 224, 224).cuda(0)
    # model = res2next50(pretrained=True)
    # model = model.cuda(0)
    # print(model(images).size())

    with fluid.dygraph.guard():
        import numpy as np
        in_np = np.ones(shape=[1, 3, 224, 224], dtype="float32")
        in_var = fluid.dygraph.to_variable(in_np)
        model = res2next50(pretrained=False)
        out = model(in_var)
        print(out.shape)
