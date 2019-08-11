# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:42:35 2019

@author: qiqi
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


class SVM(object):
    
    def __init__(self,C,kernel='rbf',sigma=0.1):
        '''C越大,sigma越小，越容易出现高方差、低偏差'''
        self.C = C
        self.kernel = kernel
        self.model = svm.SVC(C=C, kernel=kernel, gamma = np.power(sigma,-2.)/2)
        
    def Show_Data(self,x,y,show_boundary=False):
        '''show_boundary为False时此函数不用于显示边界'''
        plt.figure(figsize=(10,6))
        plt.scatter(x[:,0], x[:,1], c=y.flatten(), marker='o')
        plt.xlabel('X1')
        plt.ylabel('X2')
        if show_boundary == True:return
        plt.show()
        
    def Show_Boundary(self,x,y):
        '''画出分类边界'''
        self.Show_Data(x=x, y=y, show_boundary=True)
        '''np.linspace用于生成从最小值到最大值共1000个元素的等差数列'''
        '''np.meshgrid用于生成坐标点地图，x1与x2维度均为(1000,1000)'''
        x1, x2 = np.meshgrid(np.linspace(np.min(x[:,0]),np.max(x[:,0]),1000),
                             np.linspace(np.min(x[:,1]),np.max(x[:,1]),1000))
        
        '''np.c_将两个拥有相同行数的数组按行合并，转化后得到一个二维数组，每行有两个元素x1,x2'''
        pred = self.model.predict(np.c_[x1.flatten(),x2.flatten()])
        '''pred维度为(1000*1000,)'''
        pred = pred.reshape(x1.shape)
        '''此时x1,x2,pred维度均为(1000,1000)'''
        plt.contour(x1,x2,pred)
        
    def Train(self,x,y):
        '''对数据进行训练拟合'''
        self.model.fit(x, y.ravel())
        
            
    def Predict(self,x,y):
        '''对数据进行预测,返回result和score,result为预测数据的预测结果,是一个列表,score为准确度,是实数'''
        result = []
        for xp,yp in zip(x,y):
            result.append((self.model.predict([xp]),yp))
        return result,self.model.score(x,y)


def read_data(file_path):
    '''读取数据'''
    file = open(file_path,'r',encoding='utf-8')
    read_file = csv.reader(file)
    rows = [row for row in read_file]
    data = np.array(rows)
    x = data[1:,1:-1]
    x = x.astype(float)
    y = data[1:,-1]
    labels = np.zeros(y.shape)
    '''将类别进行编码'''
    for i in range(y.shape[0]):
        if y[i] == 'setosa':
            labels[i] = 0.0
        elif y[i] == 'versicolor':
            labels[i] = 1.0
        else:
            labels[i] = 2.0
    file.close()
    return x,labels

'''读取数据'''
x,y = read_data('iris.csv')

'''随机打乱x,y，使用np.random.get_state()和np.random.set_state保证x,y保持原来对应关系'''
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)

'''前130行数据用作训练，剩下数据用在测试'''
x_train = x[0:130]
y_train = y[0:130]
x_test = x[130:]
y_test = y[130:]

'''使用SVM进行分类'''
s = SVM(C=30, kernel='rbf', sigma=1.1)
s.Train(x_train,y_train)

'''对训练集和测试集数据进行预测,由于训练集数据较大,故这里只打印了测试集每条数据的预测结果'''
train_result, train_score = s.Predict(x=x_train, y=y_train)
test_result, test_score = s.Predict(x=x_test, y=y_test)
print('train_score:',train_score,'  test_score:',test_score)
print('test_result:')
for it in test_result:
    print('Predict:',int(it[0][0]),'  Label:',int(it[1]))