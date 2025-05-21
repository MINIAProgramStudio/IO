import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import cv2
from os import listdir, remove
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
layers = tf.keras.layers

print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

path = "data/cifar_10/data_batch_1"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = "bytes")
    return dict

cifar_10_data = unpickle(path)

cifar_10_img = cifar_10_data[b"data"]
cifar_10_labels = cifar_10_data[b"labels"]
del cifar_10_data

cifar_10_img = np.transpose(np.reshape(cifar_10_img,(len(cifar_10_img), 3, 32,32)),(0,2,3,1))
print(type(cifar_10_img[0]))
plt.imshow(cifar_10_img[0])