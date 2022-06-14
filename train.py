from model import AnomalyDetector
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
def train():
    """
    训练自编码模型
    :return:
    """
    data = np.load('.\model\sumvec.npy')
    train_data,test_data = train_test_split(data,test_size=0.2,random_state=42)
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam',loss='mae')
    history = autoencoder.fit(train_data,train_data,
                              epochs=100,
                              batch_size=512,
                              validation_data=(test_data,test_data),
                              shuffle=True)
    plt.plot(history.history['loss'],label='Train Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.legend()
    plt.show()


if __name__=='__main__':
    train()
