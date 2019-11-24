from data import *
from math import *
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import tree
from numpy.random import choice
import math
from sklearn.preprocessing import normalize

class DTC:
    def __init__(self,train_set,train_labels,depth,criterion='c'):
        if criterion == 'e': c = 'entropy'
        else: c = 'gini'
        self.clf  = tree.DecisionTreeClassifier(criterion=c,splitter='random',max_depth=depth)
        self.clf.fit(train_set,train_labels)

import numpy as np
import matplotlib.pyplot as plt

global auc
auc = [0,0,0,0,0,0,0,0,0,0]

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

class AdaBoost:
    def __init__(self, Data, NoOfClassifiers = 10,L=5000,depth=10):
        self.depth = depth
        n=int(1*L)
        self.NoOfClassifiers = NoOfClassifiers
        Data.learning_set=Data.learning_set[:L]
        Data.learning_set_labels=Data.learning_set_labels[:L]
        print ("选择",L,"张图")
        self.alpha = []
        self.models = []
        self.BoostedPredictions=[]
        self.Boost(Data,n)
        self.BoostedPredVec=[]
        self.getPredictions(Data)
        self.BoostedScore = 0
        xx = []
        yy = []
        aauc = []
        for k in range(10):#十类标签
            labels = []
            scores = []
            for i in range(L):
                if Data.learning_set_labels[i] == k:
                    labels.append(1)
                else:
                    labels.append(0)
                if self.BoostedPredictions[i] == k:
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

        for i in range(L):#计算总的Acc
            if str(self.BoostedPredictions[i]) == str(Data.learning_set_labels[i]):
                self.BoostedScore += 1.0/(L*1.0)
        print (self.BoostedScore)

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
            
        '''
        print(Data.learning_set_labels)
        print(self.BoostedPredictions)
        print(labels)
        print(scores)
        '''
        '''这里做错了,出来的居然不是个曲线，气死我了 （换方法做完后注：我靠我好像就是差一个排序吧？？作业太多了，有空再补完）
        c = []#神奇的混淆矩阵
        for i in range(10):
            c.append([0,0,0,0,0,0,0,0,0,0])
        #准确率Accuacy
        for i in range(L):#生成混淆矩阵
            #print(self.BoostedPredictions[i],Data.learning_set_labels[i])
            c[int(Data.learning_set_labels[i])][int(self.BoostedPredictions[i])] += 1
            if str(self.BoostedPredictions[i]) == str(Data.learning_set_labels[i]):
                self.BoostedScore += 1.0/(L*1.0)
        print (self.NoOfClassifiers,self.BoostedScore)
        for i in range(10):
            print(c[i])
        TPR = []
        FPR = []
        for i in range(10):
            rowsum,colsum = sum(c[i]),sum(c[r][i] for r in range(10))#rowsum=TP+FN  colsum=TP+FN
            try:
                TP = c[i][i]
                FP = colsum - TP
                FN = rowsum - TP
                TN = L-(colsum+rowsum-TP)
                #print(TP,FP,FN,TN)
                TPR.append(TP/float(TP+FN))
                FPR.append(FP/float(FP+TN))
                print('precision: %s'%(TP/float(TP+FP)), 'recall: %s' %TPR[i],'F1 Score:%s'%(2*TP/float(2*TP+FP+FN)))
                #Precision = TP/(TP+FP)     Recall = TP/(TP+FN)
            except ZeroDivisionError:
                print('precision: %s'%0,'recall: %s'%0)
        print(TPR)
        print(FPR)
        import matplotlib.pyplot as plt
        import numpy as np
        plt.scatter(FPR,TPR)
        #plt.xlim(0,1)
        plt.show()
        '''
        
    def loss(self,true_label,predicted_label):
        #损失函数
        if true_label != predicted_label:
            return 1
        else:
            return 0

    def Boost(self,Data,train_set_size):
        print ("Boosting")
        L = len(Data.learning_set)
        p = [1.0/(1.0*L) for i in range(L)]
        for j in range(self.NoOfClassifiers):
            X_train, y_train = self.genTrainSet(Data,p,train_set_size)
            clf = DTC(X_train,y_train, depth = self.depth)
            predictions = clf.clf.predict(Data.learning_set)
            error = 1.0 - clf.clf.score(Data.learning_set,Data.learning_set_labels)
            
            #不是弱分类器了
            if error > 0.5:
                self.NoOfClassifiers = j
                break
            else:
                self.models.append(clf)
                self.alpha.append(0.5*log((1-error)/(1.0*error)))
                print ("Error at ",j , error,'And its alpha ',self.alpha[j])
                for i in range(0,L):
                    if predictions[i] == Data.learning_set_labels[i]:
                        p[i] = p[i]*(1.0*sqrt(1.0/(1.0*self.alpha[j])))
                    else:
                        p[i] = p[i]*(1.0*sqrt(1.0*self.alpha[j]))
                z = sum(p)
                #归一化
                for i in range(len(p)):
                    p[i]=(1.0*p[i])/(1.0*z)

    def genTrainSet(self,Data, weights, train_set_size):
        
        #选择具有概率分布权重的训练集的子集
        indices = [i for i in range(len(Data.learning_set))]
        indexlist = [choice(indices, p=weights) for i in range(train_set_size)]
        X_train = []
        y_train = []
        for i in indexlist:
            X_train.append(Data.learning_set[i])
            y_train.append(Data.learning_set_labels[i])
        return X_train,y_train

    def getPredictions(self,Data):
        #预测结果
        for i in range(len(Data.learning_set)):
            #print(self.NoOfClassifiers)#调试用
            probvec=np.array([0 for t in range(10)],dtype='float64')
            for j in range(self.NoOfClassifiers):
                vec = (self.models[j].clf.predict_proba([Data.learning_set[i]]))[0]
                #print(vec,vec.shape,self.alpha[j])#调试用
                vec = np.array(vec,dtype='float64')
                probvec += (self.alpha[j])*vec
                #print(probvec)#调试用
            norm = np.linalg.norm(probvec)#求二范数
            poo = [probvec[t]*1.0/norm for t in range(10)]
            #print(poo)#调试用
            self.BoostedPredVec.append(poo)
            self.BoostedPredictions.append(np.argmax(poo))#直接保存第i个样本的预测数字序号
'''
D=Data()
f=open('DataWeakClassifiers.pkl','wb')
import pickle
pickle.dump(D,f,pickle.HIGHEST_PROTOCOL)
f.close()

'''
import pickle
f=open('DataWeakClassifiers.pkl','rb')
D = pickle.load(f)
f.close()
e=AdaBoost(D)
