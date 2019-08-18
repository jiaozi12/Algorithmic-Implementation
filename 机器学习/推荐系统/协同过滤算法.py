# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:36:07 2019

@author: qiqi
"""

import numpy as np
from scipy.io import loadmat


class CoordinationFit(object):
    
    def __init__(self, Y, R, Features_num=3):
        '''n_m表示电影数量,n_u表示评价电影的观众数量'''
        '''Features_num表示每部电影的特征数量,x表示每部电影的特征,这些特征将由算法自动学习到,
           学习到的特征用于评价该部电影的类别(喜剧、动作...)'''
        '''theta作为参数,用于拟合预测,与x做矩阵乘法计算'''
        self.Y = Y.astype(float)
        self.R = R.astype(float)
        self.n_m = Y.shape[0]
        self.n_u = Y.shape[1]
        self.Features_num = Features_num
        self.x = np.random.randn(self.n_m, self.Features_num)
        self.theta = np.random.randn(self.n_u, self.Features_num)
        self.u = np.zeros(self.n_m)
        
        
    def MeanNormalization(self):
        '''均值规范化,为了应对某些观众没有评价过任何一部电影的情况'''
        self.u = self.Y.sum(axis=1) / self.R.sum(axis=1)
        self.u = self.u.reshape((self.n_m, 1))
        self.Y -= self.u
        #print(self.Y)  
        
    def Cost(self, lmda=1.0):
        '''代价函数,用来计算损失值'''
        loss = np.sum(np.square((self.x.dot(self.theta.T)-self.Y))[np.where(self.R==1)])/2 + lmda*0.5*np.sum(np.power(self.theta,2)) + lmda*0.5*np.sum(np.power(self.x,2))
        #loss = np.sum(np.power((np.multiply(self.x.dot(self.theta.T),self.R)-self.Y),2))*0.5 + lmda*0.5*np.sum(np.power(self.theta,2)) + lmda*0.5*np.sum(np.power(self.x,2))
        #loss /= self.Y.shape[0]*self.Y.shape[1]
        return loss
        loss = 0.0
        for i in range(self.n_m):
            for j in range(self.n_u):
                if self.R[i][j] == 1:
                    loss += pow((self.theta[j].T.dot(self.x[i]))-self.Y[i][j], 2)
        loss *= 0.5
        for i in range(self.n_m):
            for k in range(self.Features_num):
                loss += 0.5*lmda*pow(self.x[i][k], 2)
        for j in range(self.n_u):
            for k in range(self.Features_num):
                loss += 0.5*lmda*pow(self.theta[j][k], 2)
                
        return loss
        
        
    def Train(self, learning_rate=0.001, lmda=1.0, epochs=1000, display_loss=False):
        '''learning_rate为学习率,lmda为正则化参数,epochs为迭代次数,display_loss为是否打印每轮损失值'''
        for ep in range(epochs):
            #sum1 = 0.0
            #for j in range(self.n_u):
            #    for i in range(self.n_m):
            #        if self.R[i][j] == 1:
            #            sum1 += self.theta[j].dot(self.x[i].T) - self.Y[i][j]
            #print('sum1:',sum1)
            #print((self.x.dot(self.theta.T)-self.Y).shape)
            #print(np.sum((self.x.dot(self.theta.T)-self.Y)[np.where(self.R==1)]))
            #self.x -= learning_rate * (np.sum((self.x.dot(self.theta.T)-self.Y)[np.where(self.R==1)])*self.theta + lmda*np.sum(self.x))
            #self.theta -= learning_rate * ((np.sum((self.x.dot(self.theta.T)-self.Y)[np.where(self.R==1)]))*self.x + lmda*np.sum(self.theta))
            #temp = (self.x.dot(self.theta.T) - self.Y)[np.where(self.R==1)]
            #print(temp.shape)
            #for j in range(self.n_u):
            #    for i in range(self.n_m):
            #            self.x[i] -= learning_rate * (np.sum(np.square((self.theta.dot(self.X.T)-self.Y))[np.where(self.R==1)]))
            #        temp = (self.theta[j].dot(self.x[i].T) - self.Y[i][j])[np.where(self.R==1)]
            #        self.x[i] -= learning_rate * (temp*self.theta[j] + lmda*self.x[i])
            #        temp = (self.theta[j].dot(self.x[i].T) - self.Y[i][j])[np.where(self.R[i]==1)]
            #        self.theta[j] -= learning_rate * (temp*self.x[i] + lmda*self.theta[j])
            #print(self.x.shape)
            #print((self.x.dot(self.theta.T)-self.Y).shape)
            self.x -= learning_rate * (np.multiply((self.x.dot(self.theta.T)-self.Y),self.R).dot(self.theta) + lmda*self.x)
            self.theta -= learning_rate * (np.multiply((self.x.dot(self.theta.T)-self.Y), self.R).T.dot(self.x) + lmda*self.theta)
            #for j in range(self.n_u):
            #    for i in range(self.n_m):
            #        temp = 0.0
            #        if self.R[i][j] == 1:
            #            temp = self.theta[j].dot(self.x[i].T) - self.Y[i][j]
            #        self.theta[j] -= learning_rate * (temp*self.x[i] + lmda*self.theta[j])
            if display_loss == True and (ep+1)%1000==0:
                print('Epoch:',ep+1,'\tLoss:{:.2f}'.format(self.Cost(lmda=lmda)))
                
    
    def PredictStar(self, i=0, j=0):
        '''i表示电影编号,从0开始,j表示观众编号,从0开始。预测第j位观众对第i部电影的评分'''
        pred = self.x.dot(self.theta.T) + self.u
        return pred[i][j]
    
    
    def Recommend(self, i=0, num=3):
        '''i表示电影编号,从0开始。寻找与电影i相似度最大的num部电影'''
        movies = []
        for k in range(self.n_m):
            movies.append([k, np.sum(np.power(self.x[i]-self.x[k], 2))])
        movies.sort(key=lambda x:x[1], reverse=False)
        
        '''由1开始是因为movies[0]为该电影本身'''
        return movies[1:1+num]




'''读取数据'''
mat = loadmat('movies.mat')


'''Y[i][j]代表电影i,观众j的评分(0~5,0表示未评分),R[i][j]==1表示电影i,观众j对其有评分,==0表示未评价'''
Y, R = mat['Y'], mat['R']
m = loadmat('movieParams.mat')
X = m['X']
print(X.shape)
TH = m['Theta']
lmda = 0.1
loss = np.sum(np.square((X.dot(TH.T)-Y))[np.where(R==1)])/2 + lmda*0.5*np.sum(np.power(TH,2)) + lmda*0.5*np.sum(np.power(X,2))
print('loss:',loss)

C = CoordinationFit(Y=Y, R=R, Features_num=10)
C.MeanNormalization()
C.Train(learning_rate=0.003, lmda=0.1, epochs=10000000, display_loss=True)
print('预测观众j对电影i的评分为:',int(C.PredictStar(i=4, j=4)),'\t观众j对电影i的评分实际为',Y[4][4])
movies=C.Recommend(i=1, num=2)
print('与电影1类型相似的两部电影为',movies[0][0],'  ',movies[1][0])
