#### PaddleVision：基于paddlepaddle动态图模式的图像分类模型库

##### 介绍

1. 直接迁移torchvision中部分模型，resnet，vgg，alexnet等；
2. 使用paddlepadle动态图模式，直接将torchvision中的resnet，vgg， alexnet代码转换为paddlepaddle代码；
3. 将torchvison的resnet，vgg， alexnet模型的预训练模型参数保存为paddlepaddle动态图可读格式；

##### 使用说明
1. 可以使用convert_params.py直接将pytorch模型数据转换为对应的paddlepaddle动态图数据保存；
2. 修改模型保存路径，可以直接使用对应的dygraph模型即可使用加载预训练数据；
    ~~~
   from models import resnet34
   model = resnet34(pretrained=True, root="")  # root 对应模型数据的保存路径

##### 已转换的paddlepaddle动态图版预训练模型

- alexnet: [百度网盘提取码：v82t](https://pan.baidu.com/s/13sztzmcNBu5Yr_KsftVR7Q)

- vgg系列：
    - vgg11: [百度网盘提取码：0db9](https://pan.baidu.com/s/1Jm8Utkwivdg0geC7KzHT_w)
    - vgg13: [百度网盘提取码：8xhj](https://pan.baidu.com/s/12u_f_0pyxyBfgs1q1k6Zcw)
    - vgg16: [百度网盘提取码：5avl](https://pan.baidu.com/s/1MTqPqVguNbmotTxmyUFiHw)
    - vgg19: [百度网盘提取码：rgez](https://pan.baidu.com/s/1zg7WlZyiotjHRFEpZrhRlA)
- resnet系列：
    - resnet34: [百度网盘提取码：ku58](https://pan.baidu.com/s/11SS2V0LSKppJ8pTryKVa-A)
    - resnet50: [百度网盘提取码：3t4s](https://pan.baidu.com/s/1aRGVMIPNEL6qhUzzwN2MlA)
    - resnet101: [百度网盘提取码：sg8z](https://pan.baidu.com/s/1z-B9TGB1jjDstfJBwFqG5A)
    - resnet152: [百度网盘提取码：0r0p](https://pan.baidu.com/s/1j_zCsYnLCpdCAKdoOcjcPw)
- resnest系列：
    - resnest50: [百度网盘提取码：0ubv](https://pan.baidu.com/s/14tJcvf9PUeT1J2smDUVW5g)
    - resnest101: [百度网盘提取码：42yr](https://pan.baidu.com/s/1AP7Dfkdnfl5shBA-Gn5h3A)
    - resnest200: [百度网盘提取码：r2cm](https://pan.baidu.com/s/1IAKxSmsm1wTjseQknrdqmQ)
    - resnest269: [百度网盘提取码：f2fg](https://pan.baidu.com/s/1sDg3swcZdmkyI7uClA1Cqw)
- res2net系列：
    - res2net50: [百度网盘提取码：qk3g](https://pan.baidu.com/s/1bqEhoGhvKkDwIwE_F1xGaw)
    - res2net101: [百度网盘提取码：ez0c](https://pan.baidu.com/s/1AVBZxMHfNzAOz6MEutZUzg)
    - res2next: [百度网盘提取码：my0j](https://pan.baidu.com/s/1TEeGldBqgiFqrZ6oQyMkhA)


