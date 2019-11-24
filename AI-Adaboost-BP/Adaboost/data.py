from mnist import MNIST
import numpy as np
import random
from datetime import datetime
from skimage.feature import hog
class Data:
    #加载数据并将其随机分布到测试训练集中
    def __init__(self,pixels_per_cell = (8,8),cells_per_block = (3,3),orientations=9):
        self.learning_set = []
        self.learning_set_labels = []
        self.load(pixels_per_cell,cells_per_block,orientations)

    def load(self,pixels_per_cell = (8,8),cells_per_block=(3,3),orientations=9):
        #连接数据集，各一万张
        mn = MNIST('../data')
        train_raw = mn.load_training()
        test_raw = mn.load_testing()

        print ("Loaded Raw images")

        learning_set = []
        Boom = {}
        for i in range(10):
            Boom[str(i)] = []
        for i in range(0,60000):
            Boom[str(train_raw[1][i])].append(train_raw[0][i])
        for i in range(0,10000):
            Boom[str(test_raw[1][i])].append(test_raw[0][i])
        t = datetime.now().microsecond
        random.seed(t)
        #随机分布
        [random.shuffle(Boom[str(i)]) for i in range(10)]

        #提取HOG特征
        for l in range(10):
            for i in range(0,2000):
                img =  np.array(Boom[str(l)][i])
                img.shape = (28,28)
                fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block, visualize=True)
                learning_set.append([fd,l])

        t = datetime.now().microsecond
        random.seed(t)
        #打乱数据集
        random.shuffle(learning_set)

        for i in range(20000):
            self.learning_set.append(learning_set[i][0])
            self.learning_set_labels.append(learning_set[i][1])

        self.train_set = self.learning_set[:10000]
        self.train_labels = self.learning_set_labels[:10000]
        self.test_set = self.learning_set[10000:20000]
        self.test_labels = self.learning_set_labels[10000:20000]
