import os
import cv2
import torch
import imgviz
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as ff
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class LabelProcesser:
    '''
    用于读取标签图、将前向传播后的图像转换成标签图的类
    '''

    def __init__(self, filePath='./CamVid/class_dict.csv'):
        '''
        :param filePath: str,存放不同颜色代表不同标签的文件位置
        '''
        self.colorMap = self.readColorMap(filePath)
        self.cm2lbl = self.encodeLabelPix(self.colorMap)

    @staticmethod
    def readColorMap(filePath):
        '''
        用于生成colormap
        :param filePath: str,存放不同颜色代表不同标签的文件
        :return: colorMap：list，不同种类标签按照rgb存放在此列表中
        '''
        pdLabelColor = pd.read_csv(filePath, sep=',')
        colorMap = []
        for i in range(len(pdLabelColor.index)):
            temp = pdLabelColor.iloc[i]
            color = [temp['r'], temp['g'], temp['b']]
            colorMap.append(color)
        return colorMap

    @staticmethod
    def encodeLabelPix(colorMap):
        '''
        哈希编码，使用不同数字对应不同标签
        :param colorMap: list，不同种类标签按照rgb存放在此列表中
        :return: cm2lbl，ndarray,不同数字对应不同标签，如8421504对应标签值0，即第0种东西sky
        '''
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colorMap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encodeLabelImg(self, img):
        '''
        将标签图由rgb转换成数字0-11，第一类东西是0，以此类推
        :param img: 标签图
        :return: 转换成数字0-11的标签图
        '''
        img = np.array(img, dtype='int32')
        index = (img[:, :, 0] * 256 + img[:, :, 1]) * 256 + img[:, :, 2]
        return np.array(self.cm2lbl[index], dtype='int64')

    def decodeLabelImgs(self, encodeLabels, filePaths, savePath, kindNum=12):
        '''
        批量地将测试得到的由0-11编码的标签图转换为rgb的形式的标签图并保存
        :param encodeLabels: 测试得到的结果
        :param filePaths: 测试图像的位置
        :param savePath: 保存的位置
        :param kindNum: 有多少种东西
        '''
        for encodeLabel, filePath in zip(encodeLabels, filePaths):
            decodeLabel = np.zeros(shape=encodeLabel.shape)
            decodeLabel[0 == encodeLabel] = 8421504
            for k in range(1, kindNum):
                idx = np.squeeze(np.where(self.cm2lbl == k))
                decodeLabel[k == encodeLabel] = idx
            t1 = decodeLabel % 256
            temp = decodeLabel // 256
            t2 = temp % 256
            t3 = temp // 256
            decodeLabel = np.transpose(np.stack((t1, t2, t3)).astype(np.uint8), (1, 2, 0))

            if not os.path.exists(savePath):
                os.mkdir(savePath)
            cv2.imwrite(savePath + '\\pred' + filePath[filePath.rfind('\\') + 1:], decodeLabel)

            # cv2.imshow('ss', decodeLabel)
            # cv2.waitKey()

    def decodeLabelImgs1(self, encodeLabels, originImgs, classNames, filePaths, savePath):
        '''
        与上一方法类似，生成的图像略有区别。注意如果调用此方法前已经有了同名的预测图像在savePath内，会报错，需删除之前生成的图像
        :param encodeLabels: 测试得到的结果
        :param originImgs: 原始的图像，指rgb数字为0-255之间的图像
        :param classNames: 类别的名字，比如sky等等
        :param filePaths: 测试图像的位置
        :param savePath: 保存的位置
        '''
        colorMap = np.array(self.colorMap)
        for encodeLabel, originImg, filePath in zip(encodeLabels, originImgs, filePaths):
            viz = imgviz.label2rgb(
                label=encodeLabel,
                img=imgviz.rgb2gray(originImg),
                font_size=15,
                colormap=colorMap,
                label_names=classNames,
                loc="rb",
            )
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            imgviz.io.imsave(savePath + '\\predd' + filePath[filePath.rfind('\\') + 1:], viz)


class CamvidDataset(Dataset):
    '''
    用于读取CamVid数据集的类
    '''
    lp = LabelProcesser('./CamVid/class_dict.csv')  # LabelProcesser类的对象

    def __init__(self, filePath, cropSize=None, flag=False):
        '''
        :param filePath: 数据集，数据标签图的位置
        :param cropSize: 须裁剪成的尺寸
        :param flag: 是否要输出原始图，即没有转化成tensor的0-255之间的ndarray
        '''
        if len(filePath) != 2:
            raise ValueError('同时需要图片和标签文件的路径，图片路径在前')
        self.imgFilePath = filePath[0]
        self.labelFilePath = filePath[1]

        self.imgs = self.readFile(self.imgFilePath)
        self.labels = self.readFile(self.labelFilePath)

        self.cropSize = cropSize

        self.flag = flag

    def __getitem__(self, index):
        '''
        重写父类方法，用于DataLoader
        :param index: 读取测试集和标签图片的索引
        :return: sample,字典
        '''
        img = self.imgs[index]
        label = self.labels[index]

        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.centerCrop(img, label, self.cropSize)
        if self.flag:
            img, label, oriImg = self.imgTransform1(img, label)
            sample = {'img': img, 'label': label, 'oriImg': oriImg}
        else:
            img, label = self.imgTransform(img, label)
            sample = {'img': img, 'label': label}
        return sample

    @staticmethod
    def readFile(path):
        '''
        读取文件夹下所有文件的路径
        :param path: 文件夹的位置
        :return: filePathList,list,文件夹下所有路径
        '''
        fileList = os.listdir(path)
        filePathList = [os.path.join(path, img) for img in fileList]
        filePathList.sort()
        return filePathList

    @staticmethod
    def centerCrop(img, label, cropSize):
        '''
        中心裁剪将图片裁剪成想要的大小
        :param img: 想裁剪的图片
        :param label: 对应的标签图
        :param cropSize: 想裁剪成的尺寸
        :return: img,label裁剪完成后的图片和标签
        '''
        img = ff.center_crop(img, cropSize)
        label = ff.center_crop(label, cropSize)
        return img, label

    def imgTransform(self, img, label):
        '''
        将图片和标签转换成tersor
        :param img: 图片
        :param label: 标签
        '''
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        transformImg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transformImg(img)
        label = self.lp.encodeLabelImg(label)
        label = torch.from_numpy(label)
        return img, label

    def imgTransform1(self, img, label):
        '''
        将图片标签转换成tensor，输出原始图
        :param img: 图片
        :param label: 标签
        '''
        oriImg = img
        oriImg = np.array(oriImg)
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        transformImg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transformImg(img)
        label = self.lp.encodeLabelImg(label)
        label = torch.from_numpy(label)
        return img, label, oriImg

    def __len__(self):
        '''
        包含图片的数量
        '''
        return len(self.imgs)


if __name__ == '__main__':
    trainRoot = './CamVid/train'
    trainLabel = './CamVid/train_labels'
    valRoot = './CamVid/val'
    valLabel = './CamVid/val_labels'
    testRoot = './CamVid/test'
    testLabel = './CamVid/test_labels'

    cropSize = (352, 480)

    camTrain = CamvidDataset([trainRoot, trainLabel], cropSize)
    camVal = CamvidDataset([valRoot, valLabel], cropSize)
    camTest = CamvidDataset([testRoot, testLabel], cropSize)

    trainData = DataLoader(camTrain, batch_size=2, shuffle=True, num_workers=4)
    for i, sample in enumerate(trainData):
        print('batch[%d]:' % i)
        print(sample['img'].shape)
        print(sample['label'].shape)
