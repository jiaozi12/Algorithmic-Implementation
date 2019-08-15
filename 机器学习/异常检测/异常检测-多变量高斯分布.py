# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:10:19 2019

@author: qiqi
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat


class Anomalydetection(object):
    
    def __init__(self, Epsilon=0.02):
        self.Epsilon = Epsilon
        self.Mu = None
        self.Sigma = None
        
    
    def Train(self, X_train):
        '''Mu表示每个特征的均值,为n维向量,n为数据特征数量'''
        self.Mu = np.mean(X_train, axis=0)
        self.Mu = self.Mu.reshape((1,X_train.shape[1]))
        
        '''Sigma表示每个特征的方差,为n*n矩阵,n为数据特征数量'''
        self.Sigma = np.zeros((X_train.shape[1], X_train.shape[1]))
        for i in range(X_train.shape[0]):
            self.Sigma += (X_train[i]-self.Mu).T.dot(X_train[i]-self.Mu)
        self.Sigma /= X_train.shape[0]
        
       
    def Detection(self, X):
        '''p为该样本为正常的概率,由n个特征概率连乘得到,result为检测结果'''
        p = np.ones(X.shape[0])
        result = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            p[i] = 1./(pow(2*math.pi,float(X.shape[1])/2)*pow(np.linalg.det(self.Sigma),0.5))
            p[i] *= math.exp(-0.5*(X[i]-self.Mu).dot(np.linalg.inv(self.Sigma)).dot((X[i]-self.Mu).T))
            if p[i] < self.Epsilon:
                result[i] = 1
            else:
                result[i] = 0
        return result
    
    
    def Evaluate(self, X, Y):
        '''对算法模型进行评估,返回精准率、召回率等四个指标'''
        '''TP:将实际为正样例预测为正样例的个数'''
        '''TN:将实际为负样例预测为负样例的个数'''
        '''FP:将实际为负样例预测为正样例的个数'''
        '''FN:将实际为正样例预测为负样例的个数'''
        Detection = self.Detection(X)
        TP = TN = FP = FN = 0
        for i in range(Detection.shape[0]):
            if int(Y[i]) == 1 and Detection[i] == 1:
                TP += 1
            elif int(Y[i]) == 0 and Detection[i] == 0:
                TN += 1
            elif int(Y[i]) == 0 and Detection[i] == 1:
                FP += 1
            else:
                FN += 1
        Precision = float(TP) / float(TP + FP+0.00001)
        Recall = float(TP) / float(TP + FN)
        F1_score = float(2*TP) / float(2*TP+FP+FN)
        Accuracy = float(TP+TN) / float(TP+FN+FP+TN)
        return Precision,Recall,F1_score,Accuracy
    
    
    def Plot_data(self, X):
        '''可视化某一特征,查看其是否符合高斯分布,若不符合,需进行变换转化为高斯分布。如取对数、开平方根等等'''
        plt.hist(X, 50, histtype='bar', rwidth=0.8)
        plt.xlabel('salary-group')
        plt.ylabel('salary')
        plt.title('Distribution histogram')
        plt.show()



'''读取数据'''      
mat = loadmat('data1.mat')
X_train = mat['X']
X_val, Y_val = mat['Xval'], mat['yval']

'''随机打乱x,y，使用np.random.get_state()和np.random.set_state保证x,y保持原来对应关系'''
state = np.random.get_state()
np.random.shuffle(X_val)
np.random.set_state(state)
np.random.shuffle(Y_val)

'''划分验证集和测试集'''
X_test, Y_test = X_val[0:150], Y_val[0:150]
X_val, Y_val = X_val[150:], Y_val[150:]

'''可尝试不同的Epsilon,查看验证集的F1_score,F1_score越高越好'''
Anomaly = Anomalydetection(Epsilon=0.0001)

'''查看X_train第一特征的分布直方图'''
Anomaly.Plot_data(X_train[...,0])

'''训练与评估'''
Anomaly.Train(X_train=X_train)
Precision_val, Recall_val, F1_score_val, Accuracy_val = Anomaly.Evaluate(X=X_val, Y=Y_val)
Precision_test, Recall_test, F1_score_test, Accuracy_test = Anomaly.Evaluate(X=X_test, Y=Y_test)
result = Anomaly.Detection(X_val)

'''输出评测标准'''
print('验证集')
print('Precision:{:.2f}'.format(Precision_val),'\tRecall:{:.2f}'.format(Recall_val),
      '\tF1_score:{:.2f}'.format(F1_score_val),'\tAccuracy:{:.2f}'.format(Accuracy_val))
print('测试集')
print('Precision:{:.2f}'.format(Precision_test),'\tRecall:{:.2f}'.format(Recall_test),
      '\tF1_score:{:.2f}'.format(F1_score_test),'\tAccuracy:{:.2f}'.format(Accuracy_test))