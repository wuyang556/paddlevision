# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/23
import paddle
import paddle.fluid as fluid
from paddlevision.models.utils import ReLU, ReLU6, Dropout2d
__all__ = ['MobileNetV2', 'mobilenet_v2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(fluid.dygraph.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            fluid.dygraph.Conv2D(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias_attr=False),
            fluid.dygraph.BatchNorm(out_planes),
            ReLU6()
        )


class InvertedResidual(fluid.dygraph.Layer):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            fluid.dygraph.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
            fluid.dygraph.BatchNorm(oup),
        ])
        self.conv = fluid.dygraph.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it fluid.dygraph.Sequential
        self.features = fluid.dygraph.Sequential(*features)

        # building classifier
        self.classifier = fluid.dygraph.Sequential(
            Dropout2d(0.2),
            fluid.dygraph.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, fluid.dygraph.Conv2D):
        #         fluid.dygraph.init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             fluid.dygraph.init.zeros_(m.bias)
        #     elif isinstance(m, fluid.dygraph.BatchNorm):
        #         fluid.dygraph.init.ones_(m.weight)
        #         fluid.dygraph.init.zeros_(m.bias)
        #     elif isinstance(m, fluid.dygraph.Linear):
        #         fluid.dygraph.init.normal_(m.weight, 0, 0.01)
        #         fluid.dygraph.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = fluid.dygraph.load_dygraph()
        model.set_dict(state_dict)
    return model


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = mobilenet_v2()
        state_dict = model.state_dict()
        print(state_dict.keys())
        print(len(state_dict.keys()))

        mobilenetv2_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\mobilenet_v2-b0353104.pth"
        import torch
        mobilenetv2_torch = torch.load(mobilenetv2_torch_path, map_location=torch.device("cpu"))
        print(mobilenetv2_torch.keys())
        print(len(mobilenetv2_torch.keys()))
        import torchvision
        mobilenetv2 = torchvision.models.mobilenet_v2()
        print(mobilenetv2.state_dict().keys())
        print(len(mobilenetv2.state_dict().keys()))