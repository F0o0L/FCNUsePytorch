import train
import test

if __name__ == '__main__':
    trainP = train.TrainProcesser(12)
    # trainP.train('xxx.pth')
    # trainP.reStoreParam('xxx.pth', 'xxx1.pth')

    testP=test.testPosseser(12,'xxx.pth')
    testP.test('.\\CamVid\\pred')