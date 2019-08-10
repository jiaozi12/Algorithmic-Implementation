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
        
        '''定义权重矩阵与偏置'''
        self.theta1 = np.random.randn(self.num_in,self.num_hidden)
        self.theta2 = np.random.randn(self.num_hidden,self.num_out)
        self.bias1 = np.zeros((1,self.num_hidden))
        self.bias2 = np.zeros((1,self.num_out))
        
    def Relu(self,z):
        '''定义Relu激活函数'''
        return (np.abs(z) + z) / 2.0
    
    def Sigmoid_d(self,z):
        '''定义输出函数的导数'''
        Sigmoid_d = self.Sigmoid(z)*(1-self.Sigmoid(z))
        return Sigmoid_d
    
    def Softmax(self,x,W,b,labels,lmda=1.0):
        '''计算Softmax'''
        scores = np.dot(x,W) + b
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(self.batch_size),labels])
        dscores = probs
        dscores[range(self.batch_size),labels] -= 1
        dscores /= self.batch_size
        dW = np.dot(x.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        dW += lmda*W
        return dW,db,corect_logprobs
    
    def Get_gradient(self,inputs,labels,lmda=1.0,cost=False):
        '''使用前向传播和反向传播得到梯度,cost为False此函数用于计算梯度，为True用于计算损失值'''
        m = inputs.shape[0]
        '''前向传播'''
        z2 = inputs.dot(self.theta1) + self.bias1
        a2 = self.Relu(z2)
        scores = a2.dot(self.theta2) + self.bias2
        '''将输出映射为整数'''
        exp_scores = np.exp(scores)
        '''归一化，probs每一行有num_out个元素，表示该行样本每个类别的预测概率，加和为1'''
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        '''corect_logprobs用于计算损失值'''
        corect_logprobs = np.zeros(probs.shape[0])
        for i in range(m): corect_logprobs[i] = -np.log(probs[i,int(labels[i])])
        '''scores_d为损失loss对各个分数scores的偏导数'''
        scores_d = probs
        for i in range(m): scores_d[i,int(labels[i])] -= 1
        scores_d /= m
        '''theta2_d是损失对theta2的偏导数'''
        theta2_d = np.dot(a2.T, scores_d)
        bias2_d = np.sum(scores_d, axis=0, keepdims=True)
        '''hidden_d是损失对z2的偏导数'''
        hidden_d = np.dot(scores_d, self.theta2.T)
        hidden_d[a2 <= 0] = 0
        '''theta1_d是损失对theta1的偏导数'''
        theta1_d = np.dot(inputs.T, hidden_d)
        bias1_d = np.sum(hidden_d, axis=0, keepdims=True)
        theta2_d += lmda * self.theta2
        theta1_d += lmda * self.theta1
        if cost: return corect_logprobs
        return theta1_d,theta2_d,bias1_d,bias2_d
        
    def Minimize_loss(self,inputs,labels,lmda=1.0,learning_rate=0.001):
        '''最小化损失值'''
        theta1_d,theta2_d,bias1_d,bias2_d = self.Get_gradient(inputs=inputs, labels=labels, lmda=lmda)
        self.theta1 -= learning_rate * theta1_d
        self.theta2 -= learning_rate * theta2_d
        self.bias1 -= learning_rate * bias1_d
        self.bias2 -= learning_rate * bias2_d
        
    def Cost(self,inputs,labels,lmda=1.0):
        '''计算损失值，使用MSE损失函数'''
        step = inputs.shape[0] // self.batch_size
        m = step * self.batch_size
        corect_logprobs = np.zeros(m)
        for i in range(step):
            pred = self.Get_gradient(inputs=inputs[self.batch_size*i:self.batch_size*(i+1)],
                                     labels=labels[self.batch_size*i:self.batch_size*(i+1)],
                                     cost = True)
            corect_logprobs[i*self.batch_size:(i+1)*self.batch_size] = pred
        loss = np.sum(corect_logprobs)/m + 0.5*lmda*(np.sum(self.theta1*self.theta1) + np.sum(self.theta2*self.theta2))
        return loss

        
    def Train(self,inputs,labels,learning_rate,epochs):
        '''迭代训练'''
        lmda = 1.0
        step = inputs.shape[0] // self.batch_size
        for ep in range(epochs):
            for i in range(step):
                self.Minimize_loss(inputs[self.batch_size*i:self.batch_size*(i+1)],
                                   labels[self.batch_size*i:self.batch_size*(i+1)],
                                   lmda,learning_rate)
            print('Epochs：',ep+1,'\tCost：{:.2f}'.format(self.Cost(inputs=inputs, labels=labels)),
                  '\tTrain_acc：{:.2f}'.format(self.Test(inputs=inputs, labels=labels)))
            
    def Test(self,inputs,labels,display_pred=False):
        '''测试,display_pred为True时打印预测和标签'''
        step = inputs.shape[0] // self.batch_size
        Acc = 0.0
        for i in range(step):
            hidden = self.Relu(inputs[self.batch_size*i:self.batch_size*(i+1)].dot(self.theta1) + self.bias1)
            scores = hidden.dot(self.theta2) + self.bias2
            predict = np.argmax(scores, axis=1)
            if display_pred:
                for j in range(predict.shape[0]):
                    print('Predict：',predict[j],'\tLabel：',int(labels[self.batch_size*i+j]))
            Acc += np.mean(predict == labels[self.batch_size*i:self.batch_size*(i+1)])
        Acc /= step
        return Acc
    
def read_train_data(file_path):
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
    #y = y.astype(float)
    file.close()
    return x,labels

x,y = read_train_data('iris.csv')

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

network = BP(batch_size=10, num_in=x.shape[1], num_hidden=32, num_out=3)
network.Train(inputs=x_train, labels=y_train, learning_rate=0.001, epochs=80)
print('Test_acc：{:.2f}'.format(network.Test(inputs=x_test, labels=y_test, display_pred=True)))