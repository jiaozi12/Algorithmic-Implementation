# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:02:34 2019

@author: qiqi
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat


class KMeans(object):
    
    def __init__(self, x, K=2):
        '''x为输入数据,K为聚类数量,C表示每次迭代每个点所属的类别,u表示每个聚点的坐标'''
        self.x = x
        self.K = K
        self.C = np.zeros((x.shape[0]),dtype=int)
        self.u = np.zeros((K,x.shape[1]))
        
        
    def Plot_data(self):
        '''显示未聚类的原始数据'''
        plt.figure(figsize=(10,6))
        plt.scatter(self.x[:,0], self.x[:,1], marker='o')
        plt.show()
        
        
    def Plot_color(self):
        '''显示完成聚类的数据,用不同颜色表示,这里最多可以表示len(color)中类别'''
        color = ['red','blue','green','black','yellow','pink','yellowgreen','purple','orange']
        plt.figure(figsize=(10,6))
        for i in range(self.x.shape[0]):
            plt.scatter(self.x[i,0], self.x[i,1], marker='o', c=color[self.C[i]])
        plt.show()
        
        
    def Classify(self):
        '''计算每个数据点到K个样本点的距离，并将每个数据点与最近的样本点分为一类'''
        for i in range(self.x.shape[0]):
            min_index = 0; min_dis = 1000000
            '''求与self.x[i]最近的聚点'''
            for j in range(self.K):
                dis = np.sum(np.power(self.x[i]-self.u[j],2))
                if dis < min_dis:
                    min_index = j
                    min_dis = dis
            '''将self.x[i]所属聚点类别用self.C[i]表示'''
            self.C[i] = min_index
            
                
    def Update_SamplePoints(self):
        '''更新聚点,求得每一类数据点的平均坐标作为该类新的聚点坐标'''
        for i in range(self.K):
            point = np.zeros((self.x.shape[1]))
            count = 0.0
            for j in range(self.x.shape[0]):
                if self.C[j] == i:
                    point += self.x[j]
                    count += 1
            point /= count
            '''更新聚点坐标'''
            self.u[i] = point
            
                
    def Cost(self):
        '''使用代价函数计算损失值,用于观察迭代过程'''
        sum_dis = 0.0
        for i in range(self.x.shape[0]):
            sum_dis += np.sum(np.power(self.x[i]-self.u[self.C[i]],2))
        sum_dis /= self.x.shape[0]
        return sum_dis
    
    
    def Plot_cost(self, optimal):
        '''画出每次随机初始化的损失值'''
        plt.figure(figsize=(10,6))
        op = np.arange(1,len(optimal)+1)
        loss = []
        for i in range(len(optimal)):loss.append(optimal[i][0])
        plt.plot(op, loss, linewidth=1)
        plt.xlabel("rand_num",fontsize=14)
        plt.ylabel("Loss",fontsize=14)
        plt.show()
    
        
    def Train(self, rand_num=100, epochs=100, plot_cost=False):
        '''进行迭代训练,进行多次随机初始化目的在于尽可能得到全局最优,K-Means容易陷入局部最优'''
        '''当聚类类别较少(小于等于10)时,多次初始化效果较为明显'''
        optimal = []; min_index = 0; min_cost = -1
        for rand in range(rand_num):
            '''随机生成K个不同的数字作为下标，将每个下标所对应的点作为K个样本点'''
            index = random.sample(range(0,self.x.shape[0]-1), self.K)
            for i in range(self.K):self.u[i] = self.x[index[i]]
            '''使用K-Means算法迭代epochs轮'''
            for ep in range(epochs):
                self.Classify()
                self.Update_SamplePoints()
            cost = self.Cost()
            optimal.append((cost, self.C))
            if cost < min_cost:
                min_cost = cost
                min_index = rand
        '''将多次随机初始化得到最小损失值的分类结果取出,赋值给self.C'''
        self.C = optimal[min_index][1]
        if plot_cost == False:return
        self.Plot_cost(optimal)
  
     
       
mat = loadmat('data2.mat')
x = mat['X']
k = KMeans(x,3)
k.Plot_data()
k.Train(rand_num=100, epochs=50)
k.Plot_color()
