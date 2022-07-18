from model import AnomalyDetector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import joblib
from predict import predict_threshold,print_stats
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def train(size=128):
    """
    训练自编码模型
    :return:
    """
    data = np.load('E:\github\DNS_DETECTION\DNS_DETECTION\model\sumvec' + '{}'.format(size) + '.npy')
    train_data,test_data = train_test_split(data,test_size=0.2,random_state=42)
    autoencoder = AnomalyDetector()
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    autoencoder.compile(optimizer='adam',loss='mae')
    history = autoencoder.fit(train_data,train_data,
                              epochs=100,
                              batch_size=256,
                              validation_data=(test_data,test_data),
                              shuffle=True)
    autoencoder.save_weights('E:\github\DNS_DETECTION\DNS_DETECTION\model\model_weights')
    joblib.dump(scaler,'E:\github\DNS_DETECTION\DNS_DETECTION\model\pipe.joblib')
    plt.figure()
    plt.plot(history.history['loss'],label='Train Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.legend()


    reconstructions = autoencoder.predict(train_data)
    train_loss = tf.keras.losses.mae(reconstructions,train_data)
    plt.figure()
    plt.hist(train_loss[None,:],bins=10)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()
    threshold = np.mean(train_loss) + np.std(train_loss)
    print("Threshold: ", threshold)



if __name__=='__main__':
    train()
