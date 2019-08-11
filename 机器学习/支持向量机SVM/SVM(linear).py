# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:47:32 2019

@author: qiqi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import csv


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
        self.model.fit(x, y.ravel())
        
     
mat = loadmat('data2.mat')
x = mat['X']
y = mat['y']
s = SVM(C=100, kernel='rbf', sigma=0.01)
s.Train(x,y)
s.Show_Boundary(x,y)
