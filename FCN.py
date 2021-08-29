import numpy as np
import torch
from torchvision import models
from torch import nn


class FCN(nn.Module):
    '''
    使用vgg建立FCN模型的类
    '''
    preTrainedNet = models.vgg16_bn(pretrained=True)

    def __init__(self, numClasses):
        '''
        :param numClasses: 有多少类别，在这里是12
        '''
        super().__init__()

        self.stage1 = self.preTrainedNet.features[:7]
        self.stage2 = self.preTrainedNet.features[7:14]
        self.stage3 = self.preTrainedNet.features[14:24]
        self.stage4 = self.preTrainedNet.features[24:34]
        self.stage5 = self.preTrainedNet.features[34:]

        self.score1 = nn.Conv2d(512, numClasses, 1)
        self.score2 = nn.Conv2d(512, numClasses, 1)
        self.score3 = nn.Conv2d(128, numClasses, 1)

        self.convTrans1 = nn.Conv2d(512, 256, 1)
        self.convTrans2 = nn.Conv2d(256, numClasses, 1)

        # self.upsample32x = nn.ConvTranspose2d(numClasses, numClasses, 64, 32, 16, bias=False)
        # self.upsample32x.weight.data = self.bilinearKernel(numClasses, numClasses, 64)

        # self.upsample16x = nn.ConvTranspose2d(numClasses, numClasses, 32, 16, 8, bias=False)
        # self.upsample16x.weight.data = self.bilinearKernel(numClasses, numClasses, 32)

        self.upsample8x = nn.ConvTranspose2d(numClasses, numClasses, 16, 8, 4, bias=False)
        # self.upsample8x.weight.data = self.bilinearKernel(numClasses, numClasses, 16)

        self.upsample2x1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        # self.upsample2x1.weight.data = self.bilinearKernel(512, 512, 4)

        self.upsample2x2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        # self.upsample2x2.weight.data = self.bilinearKernel(256, 256, 4)

    @staticmethod
    def bilinearKernel(inChannels, outChannels, kernelSize):
        '''
        对反卷积核初始化。感觉有问题
        :param inChannels: 反卷积层前一层的通道数
        :param outChannels: 反卷积层输出的通道数
        :param kernelSize: 卷积核的尺寸
        '''
        factor = (kernelSize + 1) // 2
        if kernelSize % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernelSize, :kernelSize]
        bilinearFilter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((inChannels, outChannels, kernelSize, kernelSize), dtype=np.float32)
        for i in range(inChannels):
            for j in range(outChannels):
                weight[i, j, :, :] = bilinearFilter
        return torch.from_numpy(weight)

    def forward(self, X):  # X:352,480,3
        '''
        前向传播
        :param X:输入的图片(图片数量，通道，高，宽)
        :return: output，前向传播的结果
        '''
        s1 = self.stage1(X)  # s1:176,240,64
        s2 = self.stage2(s1)  # s2:88,120,128
        s3 = self.stage3(s2)  # s3:44,60,256
        s4 = self.stage4(s3)  # s4:22,30,512
        s5 = self.stage5(s4)  # s5:11,15,512

        # score1 = self.score1(s5)  # score1:11,15,12
        # score1 = self.upsample32x(score1)  # score1:352,480,12

        s5 = self.upsample2x1(s5)  # s5:22,30,512
        add1 = s4 + s5  # add1:22,30,512

        # score2 = self.score2(add1)  # score2:22,30,12
        # score2 = self.upsample16x(score2)  # score2:352,480,12

        add1 = self.convTrans1(add1)  # add1:22,30,256
        add1 = self.upsample2x2(add1)  # add1:44,60,256
        add2 = add1 + s3  # add2:44,60,256

        output = self.convTrans2(add2)  # output:44,60,12
        output = self.upsample8x(output)  # output:325,480,12
        return output


if __name__ == '__main__':
    gt = np.random.rand(1, 352, 480)
    gt = gt.astype(np.int64)
    gt = torch.from_numpy(gt)
    print(gt)

    x = torch.randn(1, 3, 352, 480)
    print(x)

    net = FCN(12)
    y = net(x)
    print(y.shape)

    out = nn.functional.log_softmax(y, dim=1)
    print(out.shape)

    criterion = nn.NLLLoss()
    print(gt.shape)
    loss = criterion(y, gt)
    print(loss)
