import numpy as np
import csv

class BP(object):
    def __init__(self,batch_size, num_in, num_hidden, num_out):
        '''
        batch_size：批次大小
        num_in：输入节点个数
        num_hidden：隐藏层节点个数
        num_out：输出节点个数
        '''
        
        self.batch_size = batch_size
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        
        '''定义权重矩阵'''
        self.theta1 = np.random.randn(self.num_in,self.num_hidden)
        self.theta2 = np.random.randn(self.num_hidden,self.num_out)
        
    def Sigmoid(self,z):
        '''定义Sigmoid激活函数'''
        return .5 * (1 + np.tanh(.5 * z))
    
    def Sigmoid_d(self,z):
        '''定义输出函数的导数'''
        Sigmoid_d = self.Sigmoid(z)*(1-self.Sigmoid(z))
        return Sigmoid_d
    
    def Get_gradient(self,inputs,labels,lmda=1.0,test=False):
        '''使用前向传播和反向传播得到梯度'''
        m = inputs.shape[0]
        z2 = inputs.dot(self.theta1)
        a2 = self.Sigmoid(z2)
        z3 = a2.dot(self.theta2)
        a3 = z3
        delta3 = a3 - labels.reshape(a3.shape)
        delta2 = delta3.dot(self.theta2.T)*self.Sigmoid_d(z2)
        theta1_d = np.zeros(self.theta1.shape)
        theta1_d = (theta1_d + inputs.T.dot(delta2))/m+ lmda/m*np.sum(self.theta1)
        theta2_d = np.zeros(self.theta2.shape)
        theta2_d = (theta2_d + a2.T.dot(delta3))/m+ lmda/m*np.sum(self.theta2)
        if test:return a3
        return theta1_d,theta2_d
        
    def Minimize_loss(self,inputs,labels,lmda=1.0,learning_rate=0.001):
        '''最小化损失值'''
        theta1_d,theta2_d = self.Get_gradient(inputs,labels,lmda)
        self.theta1 = self.theta1 - learning_rate * theta1_d
        self.theta2 = self.theta2 - learning_rate * theta2_d
        
    def Cost(self,inputs,labels,lmda=1.0):
        '''计算损失值，使用MSE损失函数'''
        step = inputs.shape[0] // self.batch_size
        m = step * self.batch_size
        a3 = np.zeros((m, self.num_out))
        for i in range(step):
            pred = self.Get_gradient(inputs=inputs[self.batch_size*i:self.batch_size*(i+1)],
                                     labels=labels[self.batch_size*i:self.batch_size*(i+1)],
                                     test = True)
            a3[i*self.batch_size:(i+1)*self.batch_size] = pred
        lab = labels.reshape((labels.shape[0],1))
        J = np.sum(np.power(lab[0:m]-a3,2)) / m
        return J

        
    def Train(self,inputs,labels,learning_rate, epochs):
        '''迭代训练'''
        lmda = 1.0
        step = inputs.shape[0] // self.batch_size
        for ep in range(epochs):
            for i in range(step):
                self.Minimize_loss(inputs[self.batch_size*i:self.batch_size*(i+1)],
                                   labels[self.batch_size*i:self.batch_size*(i+1)],
                                   lmda,learning_rate)
            print('Epochs：',ep+1,'\tCost：{:.2f}'.format(self.Cost(inputs=inputs, labels=labels)))
            
    def Test(self,inputs,labels):
        '''测试，result中每个元素为1个元组，元组第一个数值表示预测，第二个表示标签，cost为测试集的损失值'''
        result = []
        step = inputs.shape[0] // self.batch_size
        for i in range(step):
            pred = self.Get_gradient(inputs=inputs[self.batch_size*i:self.batch_size*(i+1)],
                                     labels=labels[self.batch_size*i:self.batch_size*(i+1)],
                                     test = True)
            cost = self.Cost(inputs=inputs[self.batch_size*i:self.batch_size*(i+1)],
                             labels=labels[self.batch_size*i:self.batch_size*(i+1)],)
            for j in range(self.batch_size):
                result.append((pred[j][0], labels[i*self.batch_size+j]))
        return result,cost
    
def read_train_data(file_path):
    file = open(file_path,'r',encoding='utf-8')
    read_file = csv.reader(file)
    rows = [row for row in read_file]
    data = np.array(rows)
    x = data[1:,0:-1]
    x = x.astype(float)
    x_temp = x
    for i in range(x.shape[1]):
        x[...,i] = (x_temp[...,i]-np.min(x_temp[...,i])) / (np.max(x_temp[...,i])-np.min(x_temp[...,i]))
    y = data[1:,-1]
    y = y.astype(float)
    file.close()
    return x,y

x,y = read_train_data('Boston.csv')

'''随机打乱x,y，使用np.random.get_state()和np.random.set_state保证x,y保持原来对应关系'''
state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(state)
np.random.shuffle(y)

'''前450行数据用作训练，剩下数据用在测试'''
x_train = x[0:450]
y_train = y[0:450]
x_test = x[450:-1]
y_test = y[450:-1]

network = BP(batch_size=10, num_in=x.shape[1], num_hidden=32, num_out=1)
network.Train(inputs=x_train, labels=y_train, learning_rate=0.01, epochs=1000)
result,cost = network.Test(inputs=x_test, labels=y_test)
for it in result:
    print('Predict：{:.2f}\tLabel：{:.2f}'.format(it[0],it[1]))
print('Test_cost：{:.2f}'.format(cost))