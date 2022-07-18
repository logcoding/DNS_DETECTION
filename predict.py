import tensorflow as tf
from common.process import Process
from sklearn.metrics import accuracy_score,precision_score,recall_score
from model import AnomalyDetector
import numpy as np
import joblib

import matplotlib.pyplot as plt
def predict_threshold(model,data,threshold):
    """

    :param model:训练的自编码模型
    :param data: 用于预测的数据
    :param threshold: 训练选取的阈值
    :return:
    """
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions,data)
    return tf.math.less(loss,threshold)

def print_stats(predictions,labels):
    """

    :param prediction:预测的bool
    :param labels:实际的bool
    :return:
    """
    print("Accuracy = {}".format(accuracy_score(labels,predictions)))
    # print("Precision = {}".format(precision_score(labels,predictions)))
    # print("Recall = {}".format(recall_score(labels,predictions)))

def main(threshold):
    test = Process('iodine')
    test.domain_process()
    test.sumvec()
    testvec = test.wordvec
    length = len(testvec)  ###黑样本的个数
    labels = np.zeros((length,1))
    labels = np.squeeze(labels)
    test_labels = labels.astype(bool)
    # min_val = tf.reduce_min(testvec)
    # max_val = tf.reduce_max(testvec)
    #
    # test_data = (testvec - min_val) / (max_val - min_val)
    pipe = joblib.load('E:\github\DNS_DETECTION\DNS_DETECTION\model\pipe.joblib')
    test_data = pipe.transform(testvec)
    # test_data = tf.cast(test_data, tf.float32)
    predict_model = AnomalyDetector()
    predict_model.load_weights('E:\github\DNS_DETECTION\DNS_DETECTION\model\model_weights')
    preds = predict_threshold(predict_model,test_data,threshold)
    print_stats(preds,labels)
    reconstructions = predict_model.predict(test_data)
    test_loss = tf.keras.losses.mae(reconstructions,test_data)
    plt.hist(test_loss[None,:],bins=10)
    plt.xlabel("test loss")
    plt.ylabel("no of examples")
    plt.show()

if __name__=='__main__':
    threshold = 0.812
    main(threshold)





