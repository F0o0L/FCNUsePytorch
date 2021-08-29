trainRoot = './CamVid/train'  # 训练集路径
trainLabel = './CamVid/train_labels'  # 训练集标注路径
valRoot = './CamVid/val'  # 验证集路径
valLabel = './CamVid/val_labels'  # 验证集标注路径
testRoot = './CamVid/test'  # 测试集路径
testLabel = './CamVid/test_labels'  # 测试集标注路径

cropSize = (352,480)  # 输入图片需要裁剪成的尺寸
trainBatchSize = 6  # 训练时的batchSize
valBatchSize = 6  # 测试时的batchSize
epochNum = 60  # 训练多少epoch
