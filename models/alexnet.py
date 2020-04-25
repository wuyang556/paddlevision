# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/23
import paddle.fluid as fluid
from .utils import ReLU, Dropout2d


class AlexNet(fluid.dygraph.Layer):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = fluid.dygraph.container.Sequential(
            fluid.dygraph.Conv2D(3, 64, filter_size=11, stride=4, padding=2),
            ReLU(),
            fluid.dygraph.Pool2D(pool_size=3, pool_stride=2, pool_type="max"),
            fluid.dygraph.Conv2D(64, 192, filter_size=5, padding=2),
            ReLU(),
            fluid.dygraph.Pool2D(pool_size=3, pool_stride=2, pool_type="max"),
            fluid.dygraph.Conv2D(192, 384, filter_size=3, padding=1),
            ReLU(),
            fluid.dygraph.Conv2D(384, 256, filter_size=3, padding=1),
            ReLU(),
            fluid.dygraph.Conv2D(256, 256, filter_size=3, padding=1),
            ReLU(),
            fluid.dygraph.Pool2D(pool_size=3, pool_stride=2, pool_type="max"),
        )
        self.avgpool = fluid.dygraph.Pool2D(pool_size=(6, 6), global_pooling=True, pool_type="avg")
        self.classifier = fluid.dygraph.Sequential(
            Dropout2d(),
            fluid.dygraph.Linear(256 * 6 * 6, 4096),
            ReLU(),
            Dropout2d(),
            fluid.dygraph.Linear(4096, 4096),
            ReLU(),
            fluid.dygraph.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = fluid.layers.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = fluid.dygraph.load_dygraph()
        model.set_dict(state_dict)
    return model


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = alexnet()
        state_dict = model.state_dict()
        print(state_dict.keys())

        alexnet_torch_path = r"C:\Users\wuyang\.cache\torch\checkpoints\alexnet-owt-4df8aa71.pth"
        import torch
        alexnet_torch = torch.load(alexnet_torch_path, map_location=torch.device("cpu"))
        print(alexnet_torch.keys())

        new_state_dict = {}
        for key in state_dict.keys():
            print(state_dict[key].shape, alexnet_torch[key].size())
            if "classifier" in key:
                new_state_dict[key] = fluid.dygraph.to_variable(alexnet_torch[key].detach().numpy().astype("float32").transpose())
            else:
                new_state_dict[key] = fluid.dygraph.to_variable(alexnet_torch[key].detach().numpy().astype("float32"))
        print(new_state_dict.keys())

        model.set_dict(new_state_dict)

        fluid.save_dygraph(model.state_dict(), "./alexnet_from_pytorch")