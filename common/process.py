import os

import gensim.models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import dpkt
from dpkt.compat import compat_ord
# import tensorflow as tf
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
        self.wordvec = []



    def load_file(self):
        print(os.path.join('E:\github\DNS_DETECTION\DNS_DETECTION\dataset',self.file))
        if os.path.exists(os.path.join('E:\github\DNS_DETECTION\DNS_DETECTION\dataset',self.file)):
            if os.path.join('E:\github\DNS_DETECTION\DNS_DETECTION\dataset',self.file).endswith('csv'):
                dataframe = pd.read_csv(os.path.join('E:\github\DNS_DETECTION\DNS_DETECTION\dataset',self.file),index_col=0)
                data = dataframe.values
                return np.squeeze(data)
            else:
                with open(os.path.join('E:\github\DNS_DETECTION\DNS_DETECTION\dataset',self.file)) as fp:
                    data = []
                    for line in fp.readlines():
                        data.append(line)
                    return data
        else:
            print("file:{} does not exist".format(os.path.join('E:\github\DNS_DETECTION\DNS_DETECTION\dataset',self.file)))

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

    def gram2vec(self,size=128):
        """
        将n-gram数据转成向量模式
        :return: 返回n-gram词向量
        """
        models = Word2Vec(vector_size=size, window=2, min_count=2, epochs=50)
        models.build_vocab(self.gram_list)
        models.train(self.gram_list, total_examples=models.corpus_count, epochs=models.epochs)
        models.save('E:\github\DNS_DETECTION\DNS_DETECTION\model\word2vec_model_' + '{}'.format(size))




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
        print("sort_values_list:",sort_values_list)
        x = np.arange(0,len(sort_values_list))
        list_values = [val for (ind,val) in sort_values_list]
        fig,ax = plt.subplots()
        ax.plot(x,list_values)
        plt.show()

    def sumvec(self,size=128,flag=False):
        """
        对网站的n-gram向量进行求和,便于后面进行训练
        length:嵌入向量的长度大小
        """

        model = gensim.models.Word2Vec.load('E:\github\DNS_DETECTION\DNS_DETECTION\model\word2vec_model_' + '{}'.format(size))
        try:
            for line in self.gram_list:
                temp = [model.wv['{}'.format(gram)].tolist() for gram in line]
                if np.array(temp).shape[0] >= 1:
                    self.wordvec.append(list(map(sum,zip(*temp))))
        except Exception as e:
            print(e)
        self.wordvec = np.array(self.wordvec)
        if flag==True:
            np.save('E:\github\DNS_DETECTION\DNS_DETECTION\model\sumvec' + '{}'.format(size),self.wordvec)


def parse(inputfile,outfile):
    """
    对pcap文件进行解析，提取域名信息,一般用在测试数据中
    :param inputfile: 输入pcap文件名
    :param outfile: 输出文件名
    :return: 返回解析后的文件
    """
    # if not os.path.exists(".\dataset\{}".format(inputfile)):
    #     assert "not exists {}".format(inputfile)
    # if not os.path.exists(".\dataset\{}".format(outfile)):
    #     assert "not exists {}".format(outfile)
    with open(inputfile,'rb') as fin:
        with open(outfile,'w') as fout:
            pcap = dpkt.pcap.Reader(fin)
            for ts,buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    # print("Ethernet Frame:",
                    #       # mac_addr(eth.src),
                    #       # mac_addr(eth.dst),
                    #       # mac_addr(eth.type)
                    #       )
                    # print(eth.data.__class__)
                    if not isinstance(eth.data,dpkt.ip.IP):
                        print("Non IP Packet type not supported {}".format(eth.data.__class__.__name__))
                        continue
                    ip = eth.data
                    # print(ip.data.__class__)
                    if not isinstance(ip.data,dpkt.udp.UDP):
                        print("Not UDP Packet type not supported {}".format(ip.data.__class__.__name__))
                        continue
                    udp = ip.data
                    # print(udp.data.__class__,dpkt.dns.DNS.__class__)

                    dns = dpkt.dns.DNS(udp.data)
                    if dns.qr != dpkt.dns.DNS_Q:
                        continue
                    if dns.opcode != dpkt.dns.DNS_QUERY:
                        continue
                    fout.write(dns.qd[0].name + '\n')
                except Exception as e:
                    print(str(e))


if __name__=='__main__':
    parse("..\dataset\iodine002.pcap","..\dataset\iodine002")
    # A = Process('alexa.csv')
    # A.domain_process()
    # A.gram2vec()
    # model = gensim.models.Word2Vec.load('..\model\word2vec_model128')
    # A.sumvec(128)
    # train_data,test_data = train_test_split(A.wordvec,test_size=0.2,shuffle=True
    # )
    # min_val = tf.reduce_min(train_data)
    # max_val = tf.reduce_max(train_data)
    # train_data = (train_data-min_val) / (max_val - min_val)
    # test_data = (test_data-min_val) / (max_val - min_val)
    # train_data = tf.cast(train_data, tf.float32)
    # test_data = tf.cast(test_data, tf.float32)
    # plt.grid()
    # plt.plot(np.arange(128),train_data[0])
    # plt.title("A Normal Domain")
    # plt.show()









