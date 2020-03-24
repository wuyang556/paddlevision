# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/23
import paddle
import paddle.fluid as fluid


class Dropput2d(fluid.dygraph.Layer):
    def __init__(self, p=0.5):
        super(Dropput2d, self).__init__()
        self.p = p

    def forward(self, x):
        return fluid.layers.dropout(x, dropout_prob=self.p)


class ReLU(fluid.dygraph.Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return fluid.layers.relu(x)