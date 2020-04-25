"""Split-Attention"""

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
# from torch.nn.modules.utils import _pair
import paddle.fluid as fluid
from PdSeg.models.nn import ReLU


__all__ = ['SplAtConv2d']


class SplAtConv2d(fluid.dygraph.Layer):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        # padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=channels*radix, filter_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                             groups=groups*radix, bias_attr=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU()
        self.fc1 = fluid.dygraph.Conv2D(num_channels=channels, num_filters=inter_channels, filter_size=1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = fluid.dygraph.Conv2D(num_channels=inter_channels, num_filters=channels*radix, filter_size=1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            # splited = torch.split(x, channel//self.radix, dim=1)
            splited = fluid.layers.split(x, self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        # gap = F.adaptive_avg_pool2d(gap, 1)
        gap = fluid.layers.adaptive_pool2d(input=gap, pool_size=1, pool_type="avg")
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = fluid.layers.reshape(self.fc2(gap), shape=(batch, self.radix, self.channels))
        if self.radix > 1:
            # atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
            atten = fluid.layers.reshape(fluid.layers.softmax(atten, axis=1), shape=(batch, -1, 1, 1))
        else:
            # atten = F.sigmoid(atten).view(batch, -1, 1, 1)
            atten = fluid.layers.reshape(fluid.layers.sigmoid(atten), shape=(batch, -1, 1, 1))

        if self.radix > 1:
            # atten = torch.split(atten, channel//self.radix, dim=1)
            atten = fluid.layers.split(atten, num_or_sections=self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = SplAtConv2d(in_channels=512, channels=256, kernel_size=1, norm_layer=fluid.dygraph.BatchNorm)
        import numpy as np
        in_np = np.ones(shape=[4, 512, 32, 32], dtype="float32")
        in_var = fluid.dygraph.to_variable(in_np)
        out = model(in_var)
        print(out.shape)