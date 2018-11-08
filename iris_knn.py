import matplotlib.pyplot as plt
from matplotlib import figure
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

def plotIrisData(x,y):
    for t,marker,col in zip(list(range(3)),">ox","rgb"):
        plt.scatter(features[target == t, x], features[target == t, y], marker=marker, c=col)
    plt.xlabel(feature_names[x])
    plt.ylabel(feature_names[y])
'''
plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(231)
plotIrisData(0,1)
plt.subplot(232)
plotIrisData(0, 2)
plt.subplot(233)
plotIrisData(0, 3)
plt.subplot(234)
plotIrisData(1, 2)
plt.subplot(235)
plotIrisData(1, 3)
plt.subplot(236)
plotIrisData(2, 3)
plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)
'''
plt.grid(linestyle='--')

#plt.show()

print('-'*30)
combineData=np.concatenate((features,np.array([target]).T), axis=1)
print(combineData)
newfeatures=combineData[combineData[:,4] != 0,]
print(newfeatures)


def thresholdFunction(t, feature_name):
    AccuracyN = 0
    r = feature_names.index(feature_name)
    for i in newfeatures:
        if i[r] < t and i[4] == 1:
            AccuracyN += 1
        if i[r] >= t and i[4] == 2:
            AccuracyN += 1
    AccuracyRate = AccuracyN / len(newfeatures)
    return AccuracyRate


final_feature_name = ""
final_accuracy_rate = 0
final_thresold = 0
for name in feature_names:
    featuredata = newfeatures[:, feature_names.index(name)]
    for threshold in featuredata:
        AR = thresholdFunction(threshold, name)
        if AR > final_accuracy_rate:
            final_accuracy_rate = AR
            final_feature_name = name
            final_thresold = threshold

print(final_feature_name + " " + str(final_thresold) + " " + str(final_accuracy_rate))
plotIrisData(2,3)
plt.plot([0,8],[final_thresold,final_thresold],color="red",linestyle="--")
plt.text(-0.5,2,"Threthold = "+str(final_thresold) + "; " + "Accuracy rate = "+str(final_accuracy_rate))
plt.show()
