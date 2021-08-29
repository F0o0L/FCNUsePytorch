import numpy as np


def calcSemanticSegmentationConfusion(predLabels, gtLabels, classNum):
    '''
    计算混淆矩阵
    :param predLabels: 预测完的结果
    :param gtLabels: ground truth,即为标签图
    :param classNum: 类别的数量，在这里是12
    :return: confusion:混淆矩阵
    '''
    predLabels = iter(predLabels)
    gtLabels = iter(gtLabels)

    confusion = np.zeros((classNum, classNum), dtype=np.int64)
    for predLabel, gtLabel in zip(predLabels, gtLabels):
        if predLabel.ndim != 2 or gtLabel.ndim != 2:
            raise ValueError('label应该是2维')
        predLabel = predLabel.flatten()
        gtLabel = gtLabel.flatten()

        lbMax = np.max((predLabel, gtLabel))
        if lbMax >= classNum:
            expandedConfusion = np.zeros((lbMax + 1, lbMax + 1), dtype=np.int64)
            expandedConfusion[0:classNum, 0:classNum] = confusion

            classNum = lbMax + 1
            confusion = expandedConfusion

        mask = gtLabel >= 0
        tempLabel = classNum * gtLabel[mask].astype(int) + predLabel[mask]
        tempLabel = np.bincount(tempLabel, minlength=classNum ** 2)
        confusion += tempLabel.reshape((classNum, classNum))

    for iter_ in (predLabels, gtLabels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calcSemanticSegmentationIOU(confusion):
    '''
    计算IOU，交并比
    :param confusion: 混淆矩阵
    :return: iou，交并比
    '''
    iouDenominator = (confusion.sum(axis=1) + confusion.sum(axis=0)) - np.diag(confusion)
    iou = np.diag(confusion) / (iouDenominator + 1e-10)
    a = iou[:-1]
    # iou最后一个值可能为背景
    # return iou[:-1]
    return iou


def evalSemanticSegmentation(predLabels, gtLabels, classNum):
    '''
    计算一些评价指标
    :param predLabels: 预测完的结果
    :param gtLabels: ground truth,即为标签图
    :param classNum: 类别的数量，在这里是12
    :return: confusion:混淆矩阵
            iou:交并比
            miou:平均交并比
            pixelAcc:整体的准确率
            classAcc:各类别的准确率
            meanClassAcc:各类别的准确率的平均值
    '''
    confusion = calcSemanticSegmentationConfusion(predLabels, gtLabels, classNum)
    iou = calcSemanticSegmentationIOU(confusion)
    pixelAcc = np.diag(confusion) / confusion.sum()
    classAcc = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)

    return {'confusion': confusion,
            'iou': iou,
            'miou': np.nanmean(iou),
            'pixelAcc': pixelAcc,
            'classAcc': classAcc,
            'meanClassAcc': np.nanmean(classAcc)}


if __name__ == '__main__':
    predLabels = np.random.randint(0, 3, (2, 10, 10))
    gtLabels = np.random.randint(0, 3, (2, 10, 10))

    predLabels = [i for i in predLabels]
    gtLabels = [i for i in gtLabels]

    # print(predLabels)
    # print(gtLabels)

    confusion = calcSemanticSegmentationConfusion(predLabels, gtLabels, 3)
    print(confusion)

    iou = calcSemanticSegmentationIOU(confusion)
    print(iou)
