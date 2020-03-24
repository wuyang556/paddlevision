# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/17

import paddlehub
import paddle
import paddle.fluid as fluid

import torch
from .utils import ReLU, Dropput2d
# import torch.nn as nn
# import torchvision
# torchvision.models.vgg16()


class VGG(fluid.dygraph.Layer):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = fluid.dygraph.nn.Pool2D(pool_size=(7, 7), global_pooling=True, pool_type="avg")
        self.classifier = fluid.dygraph.container.Sequential(
            fluid.dygraph.Linear(512*7*7, 4096, act="relu"),
            ReLU(),
            Dropput2d(),
            fluid.dygraph.Linear(input_dim=4096, output_dim=4096, act="relu"),
            ReLU(),
            Dropput2d(),
            fluid.dygraph.Linear(input_dim=4096, output_dim=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = fluid.layers.flatten(x, axis=1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [fluid.dygraph.Pool2D(pool_size=2, pool_stride=2, pool_type="max")]
        else:
            conv2d = fluid.dygraph.Conv2D(in_channels, v, filter_size=3, padding=1)
            if batch_norm:
                layers += [conv2d,
                           fluid.dygraph.BatchNorm(num_channels=v, ),
                           ReLU()]
            else:
                # conv2d = fluid.dygraph.Conv2D(in_channels, v, filter_size=3, padding=1, act="relu")
                layers += [conv2d,
                           ReLU()]
            in_channels = v
    return fluid.dygraph.container.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = fluid.dygraph.load_dygraph()
        model.set_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs )


if __name__ == '__main__':
    from collections import OrderedDict
    import numpy as np
    with fluid.dygraph.guard():
        vgg = vgg19()
        # print(vgg.state_dict())
        print(len(vgg.state_dict().keys()))
        print(vgg.state_dict().keys())
        vgg_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\vgg19-dcbb9e9d.pth"
        vgg_torch = torch.load(vgg_torch_path, map_location=torch.device("cpu"))
        print(len(vgg_torch.keys()))
        print(vgg_torch.keys())
        # state_dict = OrderedDict()
        # for key_pd, key_torch in zip(vgg.state_dict().keys(), vgg_torch.keys()):
        #     print(vgg.state_dict()[key_pd].shape, vgg_torch[key_torch].numpy().shape)
        #     if "classifier" in key_pd and "classifier" in key_torch:
                # a = np.random.rand(2, 4)
                # a.transpose()
            #     state_dict[key_pd] = fluid.dygraph.to_variable(vgg_torch[key_torch].numpy().astype("float32").transpose())
            # else:
            #     state_dict[key_pd] = fluid.dygraph.to_variable(vgg_torch[key_torch].numpy().astype("float32"))
        # vgg.set_dict(state_dict)
        # print(vgg.state_dict())
        # fluid.dygraph.save_dygraph(state_dict=vgg.state_dict(), model_path="./vgg19")
        # fluid.dygraph.save_dygraph(state_dict, model_path="./vgg16")

