# domain = ['google.co.com','linkedin.com']
# domain_list = []
# for i in range(2):
#     temp = ''.join(domain[i].split('.')[:-1])
#     domain_list.append(temp)
# print(domain_list)
# domain_ = []
# for i in range(2):
#     temp = []
#     for j in range(len(domain_list[i])-1):
#         temp.append(domain_list[i][j:j+2])
#     domain_.append(temp)
# print(domain_)
import os
# print(os.path.join('./dataset','alexa.csv'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score
label0 = np.array([True,True,True,True,False])
label1 = np.array([True,True,True,True,True])

print(accuracy_score(label0,label1))
print(precision_score(label0,label1))
print(recall_score(label0,label1))




