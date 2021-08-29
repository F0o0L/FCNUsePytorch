import torch
import cfg
import util
import torch.nn.functional as F
from FCN import FCN
from dataset import CamvidDataset
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from evaluationSegmentation import evalSemanticSegmentation


class TrainProcesser():
    '''
    训练
    '''
    def __init__(self, classNum):
        '''
        :param classNum: 有多少类别，在这里是12
        '''
        self.classNum = classNum
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.camTrain = CamvidDataset([cfg.trainRoot, cfg.trainLabel], cfg.cropSize)
        self.camVal = CamvidDataset([cfg.valRoot, cfg.valLabel], cfg.cropSize)
        # self.camTest = CamvidDataset([cfg.testRoot, cfg.testLabel], cfg.cropSize)

        self.trainData = DataLoader(self.camTrain, batch_size=cfg.trainBatchSize, shuffle=True, num_workers=0)
        self.valData = DataLoader(self.camVal, batch_size=cfg.valBatchSize, shuffle=True, num_workers=0)

        self.fcn = FCN(self.classNum)
        self.fcn = self.fcn.to(self.device)
        self.criterion = nn.NLLLoss().to(self.device)
        self.optimizer = optim.Adam(self.fcn.parameters(), lr=1e-4)

    def train(self, savePath='xxx.pth'):
        '''
        训练
        :param savePath: 保存模型的位置
        '''
        totalPrecTime = datetime.now()
        best = [0]
        trainLosses = []
        evalLosses = []
        trainAccs = []
        evalAccs = []
        trainMIOUs = []
        evalMIOUs = []
        for epoch in range(cfg.epochNum):
            print('epoch is [{}/{}]'.format(epoch + 1, cfg.epochNum))
            if epoch % 50 == 0 and epoch != 0:
                for group in self.optimizer.param_groups:
                    group['lr'] *= 0.5

            trainLoss = 0
            trainAcc = 0
            trainMIOU = 0
            trainClassAcc = 0

            net = self.fcn.train()
            precTime = datetime.now()

            for i, sample in enumerate(self.trainData):
                trainImg = Variable(sample['img'].to(self.device))
                trainLabel = Variable(sample['label'].to(self.device))

                out = net(trainImg)
                out = F.log_softmax(out, dim=1)
                loss = self.criterion(out, trainLabel)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                trainLoss += loss.item()

                predLabel = out.max(dim=1)[1].data.cpu().numpy()
                predLabel = [j for j in predLabel]

                trueLabel = trainLabel.data.cpu().numpy()
                trueLabel = [j for j in trueLabel]

                evalMetrix = evalSemanticSegmentation(predLabel, trueLabel, self.classNum)
                trainAcc = evalMetrix['meanClassAcc'] + trainAcc
                trainMIOU = evalMetrix['miou'] + trainMIOU
                trainClassAcc = evalMetrix['classAcc'] + trainClassAcc

                print('|Epoch|:{} |Train Batch|:{}'.format(epoch + 1, i + 1))

            net = self.fcn.eval()
            evalLoss = 0
            evalAcc = 0
            evalMIOU = 0
            evalClassAcc = 0

            with torch.no_grad():
                for i, sample in enumerate(self.valData):
                    valImg = Variable(sample['img'].to(self.device))
                    valLabel = Variable(sample['label'].to(self.device))
                    # valImg = Variable(sample['img'].to(torch.device('cpu')))
                    # valLabel = Variable(sample['label'].to(torch.device('cpu')))

                    out = net(valImg)
                    out = F.log_softmax(out, dim=1)
                    loss = self.criterion(out, valLabel)

                    evalLoss += loss.item()

                    predLabel = out.max(dim=1)[1].data.cpu().numpy()
                    predLabel = [j for j in predLabel]

                    trueLabel = valLabel.data.cpu().numpy()
                    trueLabel = [j for j in trueLabel]

                    evalMetrix = evalSemanticSegmentation(predLabel, trueLabel, self.classNum)
                    evalAcc = evalMetrix['meanClassAcc'] + evalAcc
                    evalMIOU = evalMetrix['miou'] + evalMIOU
                    evalClassAcc = evalMetrix['classAcc'] + evalClassAcc

                    print('|Epoch|:{} |Eval Batch|:{}'.format(epoch + 1, i + 1))

            trainLosses.append(trainLoss / len(self.trainData))
            evalLosses.append(evalLoss / len(self.valData))
            trainAccs.append(trainAcc / len(self.trainData))
            evalAccs.append(evalAcc / len(self.valData))
            trainMIOUs.append(trainMIOU / len(self.trainData))
            evalMIOUs.append(evalMIOU / len(self.valData))
            curTime = datetime.now()
            h, remainder = divmod((curTime - precTime).seconds, 3600)
            m, s = divmod(remainder, 60)

            epochStr1 = '|Epoch|:{}\n'.format(epoch + 1)
            epochStr2 = '|Train Loss|:{:5f}\n|Train Acc|:{:5f}\n|Train Mean IOU|:{:5f}\n'.format(
                trainLoss / len(self.trainData), trainAcc / len(self.trainData), trainMIOU / len(self.trainData))
            epochStr3 = '|Eval Loss|:{:5f}\n|Eval Acc|:{:5f}\n|Eval Mean IOU|:{:5f}\n'.format(
                evalLoss / len(self.valData), evalAcc / len(self.valData), evalMIOU / len(self.valData))

            timeStr = 'Epoch Time {:f}:{:f}:{:f}'.format(h, m, s)
            print(epochStr1 + epochStr2 + epochStr3 + timeStr)

            if max(best) <= evalMIOU / len(self.valData):
                best.append(evalMIOU / len(self.valData))
                torch.save(net.state_dict(), savePath)

            util.plotLoss(epoch+1,trainLosses, evalLosses, trainAccs, evalAccs, trainMIOUs, evalMIOUs)

        totalCurTime = datetime.now()
        h, remainder = divmod((totalCurTime - totalPrecTime).seconds, 3600)
        m, s = divmod(remainder, 60)
        print('Total Time {:f}:{:f}:{:f}'.format(h, m, s))

    def reStoreParam(self, filePath, savePath):
        '''
        读取模型继续训练
        :param filePath:读取模型的位置
        :param savePath: 保存模型的位置
        '''
        self.fcn.load_state_dict(torch.load(filePath))
        self.train(savePath)


if __name__ == '__main__':
    tp = TrainProcesser(12)
    # tp.train()
    tp.reStoreParam('xxx.pth', 'xxx1.pth')
