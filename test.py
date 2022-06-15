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
a = [1,2,3]
b = np.array(a)
print(b,b.shape)
c = b[None,:]
print(c,c.shape)


data=np.random.randint(140,180,200)
data = data[:,None]

plt.hist(data, bins=10,
         # histtype='step'
         )

plt.show()
print(data.shape)


