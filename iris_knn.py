import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import imp



data=load_iris()
#print(data)
features=data['data']
print(features)
feature_names=data['feature_names']
print(feature_names)
target=data['target']
print(target)
for t,marker,col in zip(list(range(3)),">ox","rgb"):
    plt.scatter(features[target == t, 0], features[target == t, 1], marker=marker, c=col)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.grid(linestyle='--')
plt.show()
