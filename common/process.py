import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Process:
    """
    数据处理的过程，目前只对于csv格式的数据进行处理
    """
    def __init__(self,path):
        self.file = path
        self.gram_list = []   ###alexa数据中的n元祖列表 二维数组
        self.domain_dict = {}  ###n元组字典的键值对
        self.domain_num = 0   ###统计字典中所有值的总数，用以计算频率 二元组的长度为1415
        self.len_url = 0   ###统计用于实验的网站的个数，构建onehot向量


    def load_file(self):
        if os.path.exists(os.path.join('..\dataset',self.file)):
            dataframe = pd.read_csv(os.path.join('..\dataset',self.file),index_col=0)
            data = dataframe.values
            return np.squeeze(data)
        else:
            print("file:{} does not exist".format(os.path.join('..\dataset',self.file)))

    def domain_process(self,gram=2):
        """
        对每个域名根据gram个数进行划分，默认按照2个划分
        :param gram: 划分个数
        :return:
        """
        domain = self.load_file()
        self.len_url = len(domain)
        # gram_list = []
        try:
            for i in range(self.len_url):
                domain_strip = ''.join(domain[i].split('.')[:-1]).lower()
                temp = []
                for j in range(len(domain_strip)-1):
                    temp.append(domain_strip[j:j+gram])
                self.gram_list.append(temp)
        except Exception as e:
            print('{}'.format(e))

        # domain_dict = {}  ###统计各种n元组出现的次数字典
        rows = len(self.gram_list)
        for i in range(rows):
            for j in range(len(self.gram_list[i])):
                if self.gram_list[i][j] not in self.domain_dict:
                    self.domain_dict[self.gram_list[i][j]] = 1
                else:
                    self.domain_dict[self.gram_list[i][j]] += 1
        self.domain_num = sum(self.domain_dict.values())  ###统计字典中所有值的和

    def gram2onehot(self):
        """
        将n-gram数据转成onehot编码形式
        :return: 返回onehot向量
        """
        gram_mat = np.zeros((self.len_url,self.domain_num))



    def plotfrehist(self):
        """
        绘制n元组的字典中频率图形
        :return:
        """
        fre_dict = {}  ###将大数值转成频率后的字典
        for ind,val in self.domain_dict.items():
            if ind not in fre_dict:
                fre_dict[ind] = self.domain_dict[ind] / (self.domain_num * 1.0)
        sort_values_list = sorted(fre_dict.items(),key=lambda item:item[1],reverse=True)
        x = np.arange(0,len(sort_values_list))
        list_values = [val for (ind,val) in sort_values_list]
        fig,ax = plt.subplots()
        ax.plot(x,list_values)
        plt.show()






if __name__=='__main__':
    A = Process('alexa.csv')
    A.domain_process()
    print(A.domain_dict)
    print(len(A.domain_dict))
    A.plotfrehist()









