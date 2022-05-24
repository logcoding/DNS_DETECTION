import os
import pandas as pd
import numpy as np
class Process:
    """
    数据处理的过程，目前只对于csv格式的数据进行处理
    """
    def __init__(self,path):
        self.file = path
        self.gram_list = []   ###n元祖列表
        self.domain_dict = {}  ###统计字典中所有值的和


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
        num_domain = len(domain)
        # gram_list = []
        try:
            for i in range(num_domain):
                domain_strip = ''.join(domain[i].split('.')[:-1])
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
        num_sum = sum(self.domain_dict.values())  ###统计字典中所有值的和
        return num_sum


if __name__=='__main__':
    A = Process('alexa.csv')
    num = A.domain_process()
    print(num)
    print(A.domain_dict)









