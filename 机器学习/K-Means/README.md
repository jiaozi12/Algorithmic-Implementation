### KMeans
#### 算法思路
KMeans算法是一种迭代求解的聚类分析算法，其步骤是随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。聚类中心以及分配给它们的对象就代表一个聚类。每分配一个样本，聚类的聚类中心会根据聚类中现有的对象被重新计算。这个过程将不断重复直到满足某个终止条件

#### 实现效果

未聚类样本点
![image](https://github.com/jiaozi12/Algorithmic-Implementation/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/K-Means/image/%E5%8E%9F%E5%A7%8B%E5%9B%BE%E5%83%8F.png)

聚类样本点
![image](https://github.com/jiaozi12/Algorithmic-Implementation/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/K-Means/image/%E8%81%9A%E7%B1%BB%E5%90%8E.png)

详细实现见代码及注释
