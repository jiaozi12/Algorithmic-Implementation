# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:23:17 2019

@author: qiqi
"""

import numpy as np
from scipy.io import loadmat


class PCA(object):
    
    def __init__(self):
        pass
        
        
    def Normalize(self,X):
        '''对数据进行归一化处理'''
        
        '''求得每组数据的均值'''
        means = X.mean(axis=0)
        
        '''求得每组数据的标准差'''
        stds = X.std(axis=0, ddof=1)
        X_norm = (X - means) / stds
        return X_norm
    
    
    def Pca(self,k,X):
        '''对数据进行降维,U_reduce为主成分,有k列,S为对角矩阵,U_reduce和S用来恢复数据,Z为最后降维的结果'''
        sigma = X.T.dot(X) / X.shape[0]
        U, S, V = np.linalg.svd(sigma)
        U_reduce = U[...,0:k]
        Z = X.dot(U_reduce)
        return U_reduce,S,Z
    
    
    def RecoverData(self,Z,U_reduce):
        '''从降维后的数据中恢复原来数据,但并不是完全恢复'''
        X_approx = Z.dot(U_reduce.T)
        return X_approx
    
    def Evaluate(self,X,X_approx):
        '''评价降维前后数据完整性程度,返回值越接近1说明数据完整性程度越高,降维效果越好'''
        sum1 = sum2 = 0.0
        for i in range(X.shape[0]):
            sum1 += np.sum(np.power(X[i]-X_approx[i],2))
            sum2 += np.sum(np.power(X[i],2))
        sum1 /= X.shape[0]
        sum2 /= X.shape[0]
        return 1. - sum1 / sum2
            
        
'''读取数据'''      
mat = loadmat('data1.mat')
X = mat['X']

'''创建PCA实例'''
p = PCA()

'''数据归一化'''
X = p.Normalize(X)

'''降维处理'''
U_reduce, S, Z = p.Pca(k=1, X=X)

'''压缩重现'''
X_approx = p.RecoverData(Z,U_reduce)

'''评价PCA效果'''
score = p.Evaluate(X=X, X_approx = X_approx)
print(score)