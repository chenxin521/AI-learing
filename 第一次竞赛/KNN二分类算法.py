# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:56:44 2019

@author: Lenovo
"""
import numpy as np

##导入和初始化数据
def init_data():
    iris_train = np.loadtxt('HTRU_2_train.csv',delimiter=',')
    iris_test = np.loadtxt('HTRU_2_test.csv',delimiter=',')
    dataMatIn = iris_train[:,:-1]
    classLables = iris_train[:,-1]
    classLables = classLables.ravel()

    return dataMatIn,classLables,iris_test

##KNN算法
def classify(X,dataMatIn,classlabels,k):
    distances=(((dataMatIn-X)**2).sum(axis=1))**0.5
    sortedDistances=distances.argsort()
    classCounts = [[] for i in range(2)] 
    classCounts[0].append(1.0)
    classCounts[0].append(0)
    classCounts[1].append(0.0)
    classCounts[1].append(0)
    for i in range(k):
        voteIlabel=classlabels[sortedDistances[i]]
        for j in range(2):
            if classCounts[j][0] == voteIlabel:
                classCounts[j][1] = classCounts[j][1]+1
    sortedClass = sorted(classCounts, key=lambda x : x[1],reverse=True)   
    
    return sortedClass[0][0]

##对测试集进行分类    
k_neighbors = 54       
dataMatIn,classLables,iris_test = init_data()
lables_test = []
for i in iris_test:
    y_test = classify(i,dataMatIn,classLables,k_neighbors)
    lables_test.append(y_test)

##将分类结果整合成和CSV要求格式一样的加序号的二维数组
m = len(lables_test)
prediction = [[] for i in range(m)]
for i in range(m):  
        prediction[i].append(i+1)
        prediction[i].append(lables_test[i])
print(prediction)


##将分类结果的二维数组写进CSV文件
f = open(r'C:\Users\Lenovo\Desktop\\sub8.csv','w')
np.savetxt(r'C:\Users\Lenovo\Desktop\\sub8.csv',prediction,delimiter=',')
f.close()
