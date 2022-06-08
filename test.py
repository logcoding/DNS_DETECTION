import numpy as np
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
# print(19831/9757330)
#
# input_array = np.random.randint(1000,size=(32,10))
# print(input_array)
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
vocab = ['a','b','c','d']
data = tf.constant([['a','c','d'],['d','z','b']])
layer = StringLookup(vocabulary=vocab)
layer(data)

