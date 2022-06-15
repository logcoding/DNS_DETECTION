import tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score
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
    print("Precision = {}".format(precision_score(labels,predictions)))
    print("Recall = {}".format(recall_score(labels,predictions)))