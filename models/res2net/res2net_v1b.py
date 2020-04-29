
import math
import paddle
import paddle.fluid as fluid
# from ..utils import ReLU, Dropout2d
from PaddleVision.models.utils import ReLU

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']


class Bottle2neck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = fluid.dygraph.Conv2D(num_channels=inplanes, num_filters=width*scale, filter_size=1, bias_attr=False)
        self.bn1 = fluid.dygraph.BatchNorm(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = fluid.dygraph.Pool2D(pool_type="avg", pool_size=3, pool_stride=stride, pool_padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(fluid.dygraph.Conv2D(num_channels=width, num_filters=width, filter_size=3, stride=stride, padding=1, bias_attr=False))
          bns.append(fluid.dygraph.BatchNorm(width))
        self.convs = fluid.dygraph.container.LayerList(convs)
        self.bns = fluid.dygraph.container.LayerList(bns)

        self.conv3 = fluid.dygraph.Conv2D(num_channels=width*scale, num_filters=planes * self.expansion, filter_size=1, bias_attr=False)
        self.bn3 = fluid.dygraph.BatchNorm(planes * self.expansion)

        self.relu = ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # spx = torch.split(out, self.width, 1)
        # print(out.shape)
        # print(self.width)
        spx = fluid.layers.split(out, num_or_sections=out.shape[1]//self.width, dim=1)
        # print(len(spx))

        for i in range(self.nums):
          if i == 0 or self.stype == 'stage':
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
          out = fluid.layers.concat([out, spx[self.nums]],1)
        elif self.scale != 1 and self.stype=='stage':
          out = fluid.layers.concat([out, self.pool(spx[self.nums])],1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(fluid.dygraph.Layer):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(3, 32, 3, 2, 1, bias_attr=False),
            fluid.dygraph.BatchNorm(32),
            ReLU(),
            fluid.dygraph.Conv2D(32, 32, 3, 1, 1, bias_attr=False),
            fluid.dygraph.BatchNorm(32),
            ReLU(),
            fluid.dygraph.Conv2D(32, 64, 3, 1, 1, bias_attr=False)
        )
        self.bn1 = fluid.dygraph.BatchNorm(64)
        self.relu = ReLU()
        self.maxpool = fluid.dygraph.Pool2D(pool_type="max", pool_size=3, pool_stride=2, pool_padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = fluid.dygraph.Pool2D(pool_type="avg", global_pooling=True, pool_size=1)
        self.fc = fluid.dygraph.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fluid.dygraph.container.Sequential(
                fluid.dygraph.Pool2D(pool_type="avg", pool_size=stride, pool_stride=stride,
                                     ceil_mode=True),
                fluid.dygraph.Conv2D(self.inplanes, planes * block.expansion,
                                     filter_size=1, stride=1, bias_attr=False),
                fluid.dygraph.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return fluid.dygraph.container.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = fluid.layers.reshape(x, shape=[x.shape[0], -1])
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    if pretrained:
        model_path = r"E:\Code\Python\PaddleSeg\Res2Net\res2net50_v1b_26w_4s-3cf99910.pth"
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    if pretrained:
        model_path = r"E:\Code\Python\PaddleSeg\Res2Net\res2net50_v1b_26w_4s-3cf99910.pth"
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
    return model


def res2net50_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_path = r"E:\Code\Python\PaddleSeg\Res2Net\res2net50_v1b_26w_4s-3cf99910.pth"
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model_path = r"E:\Code\Python\PaddleSeg\Res2Net\res2net101_v1b_26w_4s-0812c246.pth"
        state_dict = fluid.dygraph.load_dygraph(model_path)
        model.load_dict(state_dict[0])
    return model


def res2net152_v1b_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
    return model


if __name__ == '__main__':
    # images = torch.rand(1, 3, 224, 224)
    # model = res2net50_v1b_26w_4s(pretrained=True)
    # # model = res2net101_v1b_26w_4s(pretrained=True)
    # model = model
    # print(model(images).size())
    # model_path = r"E:\Code\Python\PaddleSeg\Res2Net\res2net101_v1b_26w_4s-0812c246.pth"
    # state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    # print(state_dict.keys())

    with fluid.dygraph.guard():
        import numpy as np
        in_np = np.ones(shape=[1, 3, 224, 224], dtype="float32")
        in_var = fluid.dygraph.to_variable(in_np)
        model = res2net50_v1b()
        out = model(in_var)
        print(out.shape)
