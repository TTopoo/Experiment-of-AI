import os
import struct
from array import array

class MNIST(object):
    
    #把数据加载到test_images ,test_labels ,train_images ,train_labels
    def __init__(self, path='.'):
        
        self.path = path
        self.test_images = []
        self.test_labels = []
        self.train_images = []
        self.train_labels = []

    def load_testing(self):
    
        ims, labels = self.load(os.path.join(self.path, 't10k-images-idx3-ubyte'),
                                os.path.join(self.path, 't10k-labels-idx1-ubyte'))
        self.test_images = ims#测试数据图片
        self.test_labels = labels#测试数据标签

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, 'train-images-idx3-ubyte'),
                                os.path.join(self.path, 'train-labels-idx1-ubyte'))

        self.train_images = ims#训练数据图片
        self.train_labels = labels#训练数据标签

        return ims, labels

    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))
            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels
