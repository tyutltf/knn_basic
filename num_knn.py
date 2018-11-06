from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import imp
from sklearn.model_selection import train_test_split

data=pd.read_csv(os.getcwd()+'\data'+os.sep+'numbers.csv')
print('原始数据:\n',data)

X_train,X_test,y_train,y_test=train_test_split(data['number'],data['classes'],test_size=0.25,random_state=40)
print('训练特征值:\n',X_train.values)
print('训练标签值:\n',y_train.values)
print('测试特征值:\n',X_test.values)
print('测试标签值:\n',y_test.values)
#print(y_train)
#print(y_test)

plt.scatter(y_train,X_train)

print('创建knn模型对象...')
knn=KNeighborsClassifier(n_neighbors=3)

print('开始训练knn模型...')
knn.fit(X_train.values.reshape(len(X_train),1),y_train)
#print(X_train.values)
#print(X_train.values.reshape(len(X_train),1)) #变成列向量

# 评估函数
# 算法对象.score(测试特征值数据, 测试标签值数据)
score=knn.score(X_test.values.reshape(len(X_test),1),y_test)
print('模型训练综合得分:',score)

# 步骤6：模型预测
# predict()函数实现
# predict(新数据（二维数组类型）): 分类结果
result = knn.predict([[12],[1.5]])
print('分类预测的结果为:{0},{1}'.format(result[0],result[1]))

# 绘制测试数据点
plt.scatter(result[0], 12, color='r')
plt.scatter(result[1], 1.5, color='g')
plt.grid(linestyle='--')
plt.show()
