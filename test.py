import torch
import cfg
import torch.nn.functional as F

import dataset
import util
from evaluationSegmentation import evalSemanticSegmentation
from FCN import FCN
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import CamvidDataset


class testPosseser():
    '''
    测试
    '''
    def __init__(self, classNum, filePath='xxx.pth'):
        '''
        :param classNum: 有多少类别，在这里是12
        :param filePath: 模型的位置
        '''
        self.classNum = classNum
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.camTest = CamvidDataset([cfg.valRoot, cfg.valLabel], cfg.cropSize,True)

        self.testData = DataLoader(self.camTest, batch_size=cfg.valBatchSize, shuffle=False, num_workers=0)

        self.fcn = FCN(self.classNum)
        # print(self.fcn)
        self.fcn = self.fcn.to(self.device)
        self.criterion = nn.NLLLoss().to(self.device)
        self.fcn.load_state_dict(torch.load(filePath))

    def test(self, savePath):
        '''
        测试
        :param savePath: 保存模型的位置
        '''
        testLoss = 0
        testAcc = 0
        testMIOU = 0
        testClassAcc = 0

        idx = 0
        net = self.fcn.eval()
        with torch.no_grad():
            for i, sample in enumerate(self.testData):
                testImg = Variable(sample['img'].to(self.device))
                testLabel = Variable(sample['label'].to(self.device))
                oriImg = sample['oriImg']
                oriImg = oriImg.numpy()
                oriImg = [j for j in oriImg]

                out = net(testImg)
                out = F.log_softmax(out, dim=1)
                loss = self.criterion(out, testLabel)

                testLoss += loss.item()

                predLabel = out.max(dim=1)[1].data.cpu().numpy()
                predLabel = [j for j in predLabel]

                trueLabel = testLabel.data.cpu().numpy()
                trueLabel = [j for j in trueLabel]

                evalMetrix = evalSemanticSegmentation(predLabel, trueLabel, self.classNum)
                testAcc = evalMetrix['meanClassAcc'] + testAcc
                testMIOU = evalMetrix['miou'] + testMIOU
                testClassAcc = evalMetrix['classAcc'] + testClassAcc

                filePath = self.camTest.imgs[idx:idx + sample['img'].shape[0]]

                # self.camTest.lp.decodeLabelImgs(predLabel, filePath, savePath, self.classNum)

                classNames = (
                    'Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian',
                    'Bicyclist', 'unlabelled')
                self.camTest.lp.decodeLabelImgs1(predLabel, oriImg, classNames, filePath, savePath)

                idx += sample['img'].shape[0]

                print('|Test Batch|:{}'.format(i + 1))

        str = '|Test Loss|:{:5f}\n|Test Acc|:{:5f}\n|Test Mean IOU|:{:5f}\n'.format(
            testLoss / len(self.testData), testAcc / len(self.testData), testMIOU / len(self.testData))
        print(str)


if __name__ == '__main__':
    util.seed_torch()
    tp = testPosseser(12)
    tp.test('.\\CamVid\\pred')
