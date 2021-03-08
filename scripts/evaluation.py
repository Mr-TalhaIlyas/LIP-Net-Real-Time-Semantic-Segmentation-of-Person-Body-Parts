import tensorflow as tf
import numpy as np
from models import  num_of_classes
if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
    from keras.layers.merge import concatenate
    from keras.layers import Activation
    import keras.backend as K
    from keras.backend import squeeze
if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Activation, concatenate
    import tensorflow_addons as tfa
    from tensorflow.keras.backend import squeeze

num_class = num_of_classes()



def grass_iou(y_true, y_pred, smooth=1):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    y_true = y_true[:,:,:,2:3]#grass ch
    y_pred = y_pred[:,:,:,2:3]#grass ch
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou

def white_clvr_iou(y_true, y_pred, smooth=1):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    y_true = y_true[:,:,:,9:11]#white_clvr ch
    y_pred = y_pred[:,:,:,9:11]#white_clvr ch
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou
def red_clvr_iou(y_true, y_pred, smooth=1):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    y_true = y_true[:,:,:,11:13]#red_clvr ch
    y_pred = y_pred[:,:,:,11:13]#red_clvr ch
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou

def weed_iou(y_true, y_pred, smooth=1):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    y_true = y_true[:,:,:,6:9]#weed ch
    y_pred = y_pred[:,:,:,6:9]#weed ch
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou

def soil_iou(y_true, y_pred, smooth=1):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    y_true = y_true[:,:,:,0:1]#soil ch
    y_pred = y_pred[:,:,:,0:1]#soil ch
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou

def unc_clvr_iou(y_true, y_pred, smooth=1):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    y_true = y_true[:,:,:,13:14]#soil ch
    y_pred = y_pred[:,:,:,9:10]#soil ch
    
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou

'''
def weight_decay(model):
    print('\n ***Adding Weight Decay*** \n')
    alpha = w_decay_value / 4 # weight decay coefficent for Depthwise and Separable Convs.
    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
            layer.add_loss(l2(alpha)(layer.depthwise_kernel))
        if isinstance(layer, SeparableConv2D):
            layer.add_loss(l2(alpha)(layer.depthwise_kernel))
            layer.add_loss(l2(alpha)(layer.pointwise_kernel))
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(w_decay_value)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(l2(w_decay_value)(layer.bias))
        return model
'''

'''
###############################################
# This one still in a working in prgoress
###############################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 
import random
import tensorflow as tf
from pallet_n_classnames import pallet_ADE20K, pallet_cityscape, pallet_VOC, pallet_mine, pallet_vistas
import keras.backend as K
import re
from tqdm import tqdm
from tabulate import tabulate
import sys

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def gray2encoded(y_true, num_class, multiplier):
    num_class = num_class
    
    y_true = y_true * multiplier
    y_true = tf.cast(y_true, 'int32')
    sess1 = tf.compat.v1.Session()
    y_true = sess1.run(y_true).squeeze()
    encoded_op = tf.one_hot(y_true, num_class, axis = -1)
    sess1 = tf.compat.v1.Session()
    encoded_op = sess1.run(encoded_op)
    return encoded_op

def mean_iou(y_true, y_pred, smooth=1):
    
    y_true = y_true[np.newaxis,:,:,:]
    y_pred = y_pred[np.newaxis,:,:,:]
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    sess = tf.compat.v1.Session()
    iou = sess.run(iou)
    return iou

num_class = 15
labels = '/home/user01/data_ssd/Talha/synthetic_clover/test/masks/masks'
preds = '/home/user01/data_ssd/Talha/synthetic_clover/preds'

classes = np.array(['soil(BG)','clover','grass','weeds','white_clover','red_clover','dandelion','shepherds_purse','thistle',
                   'white_clover_flower','white_clover_leaf','red_clover_flower','red_clover_leaf','unknown_clover_leaf'
                   ,'unknown_clover_flower', 'Mean IoU']).reshape((-1,1))

labels_list = sorted(os.listdir(labels), key=numericalSort)# This will get all the file names in the folder
preds_list = sorted(os.listdir(preds), key=numericalSort)

labels_paths = [os.path.join(labels, fname) for fname in labels_list]
preds_paths = [os.path.join(preds, fname) for fname in preds_list]
temp = []
for labelspaths, predspaths in tqdm(zip(labels_paths, preds_paths), total = len(preds_paths)):
    y_true = cv2.imread(labelspaths,-1)
    y_pred = cv2.imread(predspaths,-1)

    y_true = gray2encoded(y_true, num_class=num_class, multiplier=1)
    y_pred = gray2encoded(y_pred, num_class=num_class, multiplier=1)
    temp2 = []
    for i in range(num_class):
        iou = mean_iou(y_true[:,:,i:i+1], y_pred[:,:,i:i+1], smooth=1)
        temp2.append(iou)
    temp.append(temp2)
#    
c_iou = np.asarray(temp)

m_iou = np.mean(c_iou, axis = 1)
c_m_iou = np.mean(c_iou, axis = 0).reshape(num_class, 1)
M_iou = np.mean(m_iou)
c_m_iou=np.append(c_m_iou, [M_iou]).reshape(-1,1)
catagory_iou = np.concatenate((classes, c_m_iou), 1)
tabel = tabulate(np.ndarray.tolist(catagory_iou), headers = ["Class", "IoU"], tablefmt="github")
print(tabel)

'''