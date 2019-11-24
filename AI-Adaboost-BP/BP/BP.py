import numpy as np
import random ,array
import os, struct
import matplotlib.pyplot as plt

from numpy import append, array, int8, uint8, zeros
from mnist import MNIST

class NeuralNet(object):

    # 初始化神经网络，sizes是神经网络的层数和每层神经元个数
    def __init__(self, sizes):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)  # 层数
        # w_、b_初始化为正态分布随机数
        # w_是第二、三层的神经元的初始化，一个对象是784行40列的正态分布array，另一个对象是40行10列的array
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # b_是第二、三层的神经元的偏移，40*1和10*1
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]

    def sigmoid(self, z):# 激活函数，控制每个神经元的输出在0-1范围之内。
        return 1.0 / (1.0 + np.exp(-z))

    # Sigmoid函数的导函数，用在反向传播误差，更新权重
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # 前馈，根据输入的x，进行一次迭代，返回结果是列表，里面包含输出层神经元应该输出的结果，
    # 每层每个神经元使用下面的公式计算输出，然后再把上一层的输出值赋给x继续迭代。
    def feedforward(self, x):
        for b, w in zip(self.b_, self.w_):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    # 误差反向传播更新权值矩阵。x是输入向量，y是标签
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.b_, self.w_):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers_):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)
    
    # 学,更新权重
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            
        self.w_ = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.b_, nabla_b)]

    # training_data是训练数据(x,y);epochs是训练次数;mini_batch_size是每次训练样本数;eta是learning rate
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j+1))

    # 对模型进行评估
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # 返回output_activations和y之差
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    def predict(self, data):# 预测
        value = self.feedforward(data)
        return value.tolist().index(max(value))


def load_samples(dataset="training_data"):
    mn = MNIST('../data')
    if dataset=='training_data':
        image, label = mn.load_training()
    else:
        image, label = mn.load_testing()

    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    X = [x / 255.0 for x in X]  # 灰度化

    # 5 -> [0,0,0,0,0,1.0,0,0,0];  1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))#10*1矩阵
        e[y] = 1.0
        return e

    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    else:
        pair = list(zip(X, label))
        return pair

class Performance:
    #定义一个类，用来分类器的性能度量
    def __init__(self, labels, scores, threshold=0.5):
        #labels:数组类型，真实的标签 scores:数组类型，分类器的得分 param threshold:检测阈值
        self.labels = labels
        self.scores = scores
        self.threshold = threshold
        self.db = self.get_db()
        self.TP, self.FP, self.FN, self.TN = self.get_confusion_matrix()
    def accuracy(self):#正确率
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)
    def presision(self):#准确率
        return self.TP / (self.TP + self.FP)
    def recall(self):#召回率
        return self.TP / (self.TP + self.FN)
    def _auc(self):#auc值
        auc = 0.
        prev_x = 0
        xy_arr = self.roc_coord()
        for x, y in xy_arr:
            if x != prev_x:
                auc += (x - prev_x) * y
                prev_x = x
        return auc
    def roc_coord(self):#roc坐标
        xy_arr = []
        tp, fp = 0., 0.
        neg = self.TN + self.FP
        pos = self.TP + self.FN
        for i in range(len(self.db)):
            tp += self.db[i][0]
            fp += 1 - self.db[i][0]
            xy_arr.append([fp / neg, tp / pos])
        return xy_arr
    def roc_plot(self):#传参能手
        auc = self._auc()
        xy_arr = self.roc_coord()
        x = [_v[0] for _v in xy_arr]
        y = [_v[1] for _v in xy_arr]
        #print(x)
        #print(y)
        '''plt.title("ROC curve (AUC = %.4f)" % auc)
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.plot(x, y)
        plt.show()'''
        return x,y,auc
        
    def get_db(self):
        db = []
        for i in range(len(self.labels)):
            db.append([self.labels[i], self.scores[i]])
        db = sorted(db, key=lambda x: x[1], reverse=True)
        #print('db',db)#调试用
        return db
    def get_confusion_matrix(self):#计算混淆矩阵
        tp, fp, fn, tn = 0., 0., 0., 0.
        for i in range(len(self.labels)):
            if self.labels[i] == 1 and self.scores[i] >= self.threshold:
                tp += 1
            elif self.labels[i] == 0 and self.scores[i] >= self.threshold:
                fp += 1
            elif self.labels[i] == 1 and self.scores[i] < self.threshold:
                fn += 1
            else:
                tn += 1
        #print(tp,fp,fn,tn)#调试用
        return [tp, fp, fn, tn]

print("Training")
INPUT = 28 * 28
OUTPUT = 10
#传进去的是一个链表，分别表示每一层的神经元个数，而链表的长度刚好就是神经网络的层数
net = NeuralNet([INPUT, 40, OUTPUT])

train_set = load_samples(dataset='training_data')
test_set = load_samples(dataset='testing_data')

#training_data, epochs, mini_batch_size, eta, test_data=None):
net.SGD(train_set, 10, 100, 1.0, test_data=test_set)


# 准确率
correct = 0;
for test_feature in test_set:
    if net.predict(test_feature[0]) == test_feature[1]:
        correct += 1
print("准确率: ", correct / len(test_set))

xx = []
yy = []
aauc = []
for k in range(10):#十类标签
    labels = []
    scores = []
    #for i in range(L):
    for test_feature in test_set:
        
        if test_feature[1] == k :
            labels.append(1)
        else:
            labels.append(0)
        if net.predict(test_feature[0]) == k:
            scores.append(1)
        else:
            scores.append(0)
        p = Performance(labels, scores)
    acc = p.accuracy()
    pre = p.presision()
    rec = p.recall()
    print('数字',k,' accuracy: %.4f'% acc)
    print('数字',k,' precision: %.4f'% pre)
    print('数字',k,' recall: %.4f'% rec)
    x,y,auc=p.roc_plot()
    xx.append(x)
    yy.append(y)
    aauc.append(auc)
colors = ['aqua', 'darkorange', 'cornflowerblue', 'ghostwhite', 'indigo', 'lightblue', 'maroon', 'midnightblue', 'orchid', 'powderblue']
for i, color in zip(range(10), colors):
    plt.plot(xx[i], yy[i], color=color, lw=2,label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, aauc[i]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
