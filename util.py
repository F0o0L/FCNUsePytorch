import os
import torch
import random
import cv2
import numpy as np
import albumentations as al
import matplotlib.pyplot as plt
from dataset import CamvidDataset


def seed_torch(seed=1029):
    '''
    设置随机数种子，为使测试时每次所得结果相同
    :param seed: 种子的值
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plotLoss(epoch, trainLoss, valLoss, trainAcc, valAcc, trainMIOU, valMIOU):
    '''
    画图，随着epoch的改变画图
    :param epoch: epoch
    :param trainLoss: trainLoss
    :param valLoss: valLoss
    :param trainAcc: trainAcc
    :param valAcc: valAcc
    :param trainMIOU: trainMIOU
    :param valMIOU: valMIOU
    '''
    f, axes = plt.subplots(2, 2)
    f.set_size_inches(16, 12)
    axes[0, 0].set_title('Loss')
    axes[0, 0].plot(trainLoss, color='red', label='train loss')
    axes[0, 0].plot(valLoss, color='blue', label='val loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 1].set_title('Acc')
    axes[0, 1].plot(trainAcc, color='red', label='train acc')
    axes[0, 1].plot(valAcc, color='blue', label='val acc')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('acc')
    axes[1, 0].set_title('MIOU')
    axes[1, 0].plot(trainMIOU, color='red', label='train miou')
    axes[1, 0].plot(valMIOU, color='blue', label='val miou')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('miou')
    # plt.show()
    plt.savefig('.\\CamVid\\lossfigs\\' + str(epoch) + '.png')
    plt.close()


def scaleImg(filePath, savePath, h, w):
    '''
    缩放图片
    :param filePath: 图片所在文件夹
    :param savePath: 图片存放文件夹
    :param h: 想裁剪成的高
    :param w: 想裁剪成的宽
    '''
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    imgPathList = CamvidDataset.readFile(filePath)
    i = 0
    for imgPath in imgPathList:
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (w, h))
        cv2.imwrite(savePath + '\\' + imgPath[imgPath.rfind('\\') + 1:], img)
        i += 1
        print('第%d张图片处理完成' % i)


class DataAugmentation:
    '''
    数据增强
    '''
    def __init__(self, filePath1, filePath2, savePath1, savePath2):
        '''
        :param filePath1: 想处理的图片所在文件夹
        :param filePath2: 想处理的标签所在文件夹
        :param savePath1: 处理完后的图片所在文件夹
        :param savePath2: 处理完后的标签所在文件夹
        '''
        self.filePath1 = filePath1
        self.filePath2 = filePath2
        self.savePath1 = savePath1
        self.savePath2 = savePath2
        if not os.path.exists(self.savePath1):
            os.mkdir(self.savePath1)
        if not os.path.exists(self.savePath2):
            os.mkdir(self.savePath2)
        self.imgPathList1 = CamvidDataset.readFile(self.filePath1)
        self.imgPathList2 = CamvidDataset.readFile(self.filePath2)

    def myHorizontalFlip(self):
        '''
        镜像
        '''
        i = 0
        for imgPath1, imgPath2 in zip(self.imgPathList1, self.imgPathList2):
            img1 = cv2.imread(imgPath1)
            img2 = cv2.imread(imgPath2)

            img = al.HorizontalFlip(p=1)(image=img1, mask=img2)
            img1 = img['image']
            img2 = img['mask']

            str1 = imgPath1[imgPath1.rfind('\\') + 1:]
            str1 = str1[:-4] + 'h.png'
            str2 = imgPath2[imgPath2.rfind('\\') + 1:]
            str2 = str2[0:str2.rfind('_L')] + 'h_L.png'

            cv2.imwrite(self.savePath1 + '\\' + str1, img1)
            cv2.imwrite(self.savePath2 + '\\' + str2, img2)
            i += 1
            print('第%d张图片处理完成' % i)

    def myCrop(self, h, w):
        '''
        裁剪，这里是将一张大的图片分成左上左下右上右下四块
        :param h: 裁剪得到的图片的高
        :param w: 裁剪得到的图片的宽
        '''
        i = 1
        for imgPath1, imgPath2 in zip(self.imgPathList1, self.imgPathList2):
            img1 = cv2.imread(imgPath1)
            img2 = cv2.imread(imgPath2)

            imgH, imgW, _ = img1.shape

            crop1 = al.Crop(0, 0, w, h)(image=img1, mask=img2)
            cropImg1 = crop1['image']
            cropMask1 = crop1['mask']
            strImg1 = imgPath1[imgPath1.rfind('\\') + 1:]
            strImg1 = strImg1[:-4] + '_' + str(1) + '.png'
            strMask1 = imgPath2[imgPath2.rfind('\\') + 1:]
            strMask1 = strMask1[0:strMask1.rfind('_L')] + '_' + str(1) + '_L.png'
            cv2.imwrite(self.savePath1 + '\\' + strImg1, cropImg1)
            cv2.imwrite(self.savePath2 + '\\' + strMask1, cropMask1)
            print('第%d张图片,第%d个切割，处理完成' % (i, 1))

            crop2 = al.Crop(0, (imgH - h), w, imgH)(image=img1, mask=img2)
            cropImg2 = crop2['image']
            cropMask2 = crop2['mask']
            strImg2 = imgPath1[imgPath1.rfind('\\') + 1:]
            strImg2 = strImg2[:-4] + '_' + str(2) + '.png'
            strMask2 = imgPath2[imgPath2.rfind('\\') + 1:]
            strMask2 = strMask2[0:strMask2.rfind('_L')] + '_' + str(2) + '_L.png'
            cv2.imwrite(self.savePath1 + '\\' + strImg2, cropImg2)
            cv2.imwrite(self.savePath2 + '\\' + strMask2, cropMask2)
            print('第%d张图片,第%d个切割，处理完成' % (i, 2))

            crop3 = al.Crop(imgW - w, 0, imgW, h)(image=img1, mask=img2)
            cropImg3 = crop3['image']
            cropMask3 = crop3['mask']
            strImg3 = imgPath1[imgPath1.rfind('\\') + 1:]
            strImg3 = strImg3[:-4] + '_' + str(3) + '.png'
            strMask3 = imgPath2[imgPath2.rfind('\\') + 1:]
            strMask3 = strMask3[0:strMask3.rfind('_L')] + '_' + str(3) + '_L.png'
            cv2.imwrite(self.savePath1 + '\\' + strImg3, cropImg3)
            cv2.imwrite(self.savePath2 + '\\' + strMask3, cropMask3)
            print('第%d张图片,第%d个切割，处理完成' % (i, 3))

            crop4 = al.Crop(imgW - w, imgH - h, imgW, imgH)(image=img1, mask=img2)
            cropImg4 = crop4['image']
            cropMask4 = crop4['mask']
            strImg4 = imgPath1[imgPath1.rfind('\\') + 1:]
            strImg4 = strImg4[:-4] + '_' + str(4) + '.png'
            strMask4 = imgPath2[imgPath2.rfind('\\') + 1:]
            strMask4 = strMask4[0:strMask4.rfind('_L')] + '_' + str(4) + '_L.png'
            cv2.imwrite(self.savePath1 + '\\' + strImg4, cropImg4)
            cv2.imwrite(self.savePath2 + '\\' + strMask4, cropMask4)
            print('第%d张图片,第%d个切割，处理完成' % (i, 4))
            i += 1
            # i += 1
            # for j in range(num):
            #     r1 = np.random.randint(0, imgH - h)
            #     r2 = np.random.randint(0, imgW - w)
            #
            #     img = al.Crop(r2, r1, r2 + w, r1 + h)(image=img1, mask=img2)
            #     tempImg = img['image']
            #     tempMask = img['mask']
            #
            #     str1 = imgPath1[imgPath1.rfind('\\') + 1:]
            #     str1 = str1[:-4] + '_' + str(j) + '.png'
            #     str2 = imgPath2[imgPath2.rfind('\\') + 1:]
            #     str2 = str2[0:str2.rfind('_L')] + '_' + str(j) + '_L.png'
            #
            #     cv2.imwrite(self.savePath1 + '\\' + str1, tempImg)
            #     cv2.imwrite(self.savePath2 + '\\' + str2, tempMask)
            #
            #     print('第%d张图片,第%d个切割，处理完成' % (i, j + 1))


if __name__ == '__main__':
    # scaleImg('D:\\BaiduNetdiskDownload\\Dataset\\test_labels', 'F:\\develop\\Workplace\\myFCN\\CamVid\\test_labels',
    #          192, 256)
    # da = DataAugmentation('D:\\BaiduNetdiskDownload\\Dataset1\\train',
    #                       'D:\\BaiduNetdiskDownload\\Dataset1\\train_labels',
    #                       'D:\\BaiduNetdiskDownload\\Dataset2\\train',
    #                       'D:\\BaiduNetdiskDownload\\Dataset2\\train_labels', )
    # # da.myHorizontalFlip()
    # da.myCrop(192, 256)
    # for i in range(5):
    i = 1
    plotLoss(i, [0.1 * i, 0.2, 0.3, 0.1], [0.1, 0.3, 0.6, 0.2], [1, 2], [2, 3], [2], [3])
    # scaleImg('F:\\develop\\Workplace\\myFCN\\CamVid\\train1','F:\\develop\\Workplace\\myFCN\\CamVid',360,480)
