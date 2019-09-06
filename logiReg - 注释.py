# 逻辑回归是经典二分类算法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
# 读取数据
path = 'LogiReg_data.txt'
# 原文件没有表头，加一个表头：'Exam1','Exam2','Admitted'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
##print(pdData.head(5))
##print(pdData.shape)

# 观察数据
def data_scatter():
    positive = pdData[pdData['Admitted'] == 1]# 录取
    negative = pdData[pdData['Admitted'] == 0]# 未录取
    fig, ax = plt.subplots(figsize=(10, 5))# fig表示窗口，ax表示坐标轴
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
    # 散布点，c='b'表示蓝色。 marker='x'表示标记  label左为左上角的标签
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    # 绘图
    plt.show()
##data_scatter()
    
'''
目标：建立分类器（求解出三个参数 θ0\θ1\θ2）
设定阈值，根据阈值判断录取结果，大于0.5录取，小于则不录取
要完成的模块:
    sigmoid : 映射到概率的函数
    model : 返回预测结果值
    cost : 根据参数计算损失
    gradient : 计算每个参数的梯度方向
    descent : 进行参数更新
    accuracy: 计算精度
'''

# ***定义sigmoid函数,sigmoid 函数是将预测值（比如线性回归中的结果）,映射为概率的一个函数。
# g(z)如下
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
'''
#  可以画sigmoid函数图形验证——g:R→[0,1]  g(0)=0.5  g(−∞)=0  g(+∞)=1
# creates a vector containing 20 equally spaced values from -10 to 10
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
# 绘图
plt.show()
'''

# ***定义函数model——预测模型    h(x)=g(θTx)=g[θ0*1+θ1*x1+θ2*x2]
# X 是样本数据，每行分别为样本，每列为样本特征。
# θ 是参数，它是通过机器学习获得的，每个特征对应一个θ
# 在第0列插入一列名称为"Ones"，数值为1，方便对θ0进行运算
pdData.insert(0, 'Ones', 1)
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))#前X后theta转置为列，变为n行1列的数据
# 设置X：训练数据、y：目标值，并将数据的pandas表示形式转换为对进一步计算有用的矩阵
orig_data = pdData.values#转矩阵，也可以用.as_matrix,过时警告
X = orig_data[:,0:3]
y = orig_data[:,3:4]
theta = np.zeros([1,3])# 初始化theta

# ***定义损失函数cost——将对数似然函数,乘以一个负号,为了将求解梯度上升转换为求解梯度下降
# D(h(x),y)=-ylog(h(x))-(1-y)log(1-h(x))      J(θ)=(∑D(h(xi),yi))/n
# 这是整体损失,但样本量不同总损失不同,为统一使用平均损失,即总损失除以样本数
def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply((1 - y), np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))
# 将数值带入计算损失值
##print(cost(X, y, theta))

# ***定义梯度gradient——寻找极值,确定损失函数如何进行优化,使损失函数的值越来越小
# J对θj求导——(-∑(yi-h(xi))xij)   参数更新策略为θj=θj-α(J对θj求导)xij  j表X的j列
# 计算每个参数的梯度方向
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta) - y).ravel()#error是2维数组故要拉平,θ同理
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)
    return grad

##我们需要通过迭代来计算梯度，需要设置三种停止策略：
##    1 设置固定的迭代次数
##    2 设置损失函数的阈值，当达到一定阈值时，就停止迭代。
##    3 通过梯度的变化率来判断：设置前后两次梯度相差的阈值，如果小于该阈值，停止迭代。
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
def stopCriterion(type, value, threshold):
    # 设定三种停止策略
    if type == STOP_ITER:   return value > threshold
    elif type == STOP_COST: return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshold
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        
import numpy.random
# 洗牌
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y

import time
# 定义参数更新函数——看下时间对结果的影响
def descent(data, theta, batchSize, stopType, thresh, alpha):
    #  梯度下降
    init_time = time.time()
    i = 0 # 迭代次数
    k = 0 # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape) # 计算的梯度
    costs = [cost(X, y, theta) ] # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        
        k += batchSize # 取batch数量个数据
        if k >= n:     # 这个n是在运行的时候指定的，为取样的个数
            k = 0
            X, y = shuffleData(data) # 重新洗牌
        theta = theta - alpha * grad # 参数更新
        costs.append(cost(X, y, theta)) # 计算新的损失
        i += 1

        if stopType == STOP_ITER:       value = i
        elif stopType == STOP_COST:     value = costs
        elif stopType == STOP_GRAD:     value = grad
        if stopCriterion(stopType, value, thresh):  break
    return theta, i - 1, costs, grad, time.time() - init_time

# 根据参数选择梯度下降处理方式和停止策略
def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # import pdb; pdb.set_trace();求解——核心代码
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)

    # 在画图上显示标题等数据
    name = "Original" if (data[:,1] > 2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    
    if batchSize == n:  strDescType = "Gradient"
    elif batchSize == 1:    strDescType = "Stochastic"
    else:   strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    
    if stopType == STOP_ITER:   strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else:   strStop = "gradient norm < {}".format(thresh)
    name += strStop
    
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta

'''
终止迭代的方式有三种，而选择样本的方式同样有三种：
    1 批量梯度下降，也就是考虑所有的样本，这样的话，速度慢，但是容易得到最优解；
    2 随机梯度下降，每次只利用一个样本，这样的方式迭代速度很快，不过难以保证每次的迭代都是朝着收敛的方向；
    3 小批量梯度下降，即mini-batch，每次更新选择一小部分，比如32/64/128个样本等，这样的方式很实用，但应该先对数据进行洗牌，打乱顺序。
'''

n = 100
'''
# 选择的梯度下降方法是基于所有样本的,下面是三种停止的方式
# 1 根据固定迭代次数停止，设定为迭代5k次
    # stop_type为stop_iter,指定迭代次数的参数是thresh=5000,学习率(步长)是alpha=0.000001.
    # 步长过大有可能在取值的时候越过真实值而进行错误迭代，一般1w数据以内可先用0.01
##runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

# 2 根据损失值停止
    # stop_type为stop_cost,不人为设置迭代次数，设定阈值 1E-6, 大约需要11w次迭代
##runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)

# 3 根据梯度值停止
    #stop_type为stop_grad,设定阈值，大约迭代4w次
##runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)
'''

'''
# 开始对梯度下降的方法进行讨论和选择：
##runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.000001)
#   随机取样波动非常大，完全不收敛，学习率也过大
##runExpe(orig_data, theta, 16, STOP_ITER, thresh=5000, alpha=0.001)
#   随机小批量取样，虽然样本大了，但是浮动仍然大
# 最终考虑对数据标准化，将数据按其属性减均值，然后除方差
# 最后得到的结果是对每个属性/每列来说所有数据都聚集在0附近，方差为1
'''
from sklearn import preprocessing as pp
def data_standardize(data):
    scale_data = data.copy()
    scale_data[:, 1:3] = pp.scale(data[:, 1:3])
    return scale_data
scaled_data=data_standardize(orig_data)

##runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)
# 迭代次数越多损失下降越多，更精确
##theta = runExpe(scaled_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
# 随机梯度下降更快，但是需要的迭代次数也需要越多，还是用batch比较合适

theta = runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002*2, alpha=0.001)

# 定义精度函数
def predict(data_adj, theta):
    scaled_X = data_adj[:, :3]
    y = data_adj[:, 3]
    predictions = [1 if p >= 0.5 else 0 for p in model(scaled_X, theta)]
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))

predict(scaled_data,theta)
