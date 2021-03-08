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

'''
  1.  In this on the fly lossses the ground truths are converted on they fly into the categorical type and only 
      hybrid and tri brid losses are doing that if you wann use only one loss then convert them first
  2. ALso read DATA set guidelines in MUST READ file
'''
#%%
num_class = num_of_classes()

def dice_coef(y_true, y_pred, smooth=1):
    
    y_true = y_true * (num_class + 1) 
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=[0])
    return dice
'''
In case of binary IoU both functions below work exactly the same 
    i.e. the number of op_channel == 1
'''
def mean_iou(y_true, y_pred, smooth=1):
    
    if num_class < 2:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    else:
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    
    y_true = y_true * 255#(num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou


def binary_iou(y_true, y_pred, smooth=1):
    
    if num_class < 2:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    else:
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        
    y_true = y_true * (num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
    
    y_true = tf.cast(y_true, "int32")    
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    
    return iou
#%%
def recall_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F_Measure(y_true, y_pred):
    
    if num_class < 2:
        y_pred = tf.keras.activations.sigmoid(y_pred)
    else:
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    
    y_true = y_true * (num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
    
    y_true = tf.cast(y_true, "float32")    
    y_pred = tf.cast(y_pred, "float32")
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#%%
def tf_AUC(y_true, y_pred):
    
    y_true = y_true * (num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    y_true = y_true[:,:,:,0:1]
    
    y_true = tf.cast(y_true, "float32")    
    y_pred = tf.cast(y_pred, "float32")
    
    loss = auc_roc(y_true, y_pred)
    
    return loss
    
    
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

#%%
def dice_score(y_true, y_pred):
    
    y_true = y_true * num_class  
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth)
    return score

#%%
def Mean_IOU(y_true, y_pred):
    
    y_pred = tf.keras.activations.softmax(y_pred)
    y_true = y_true * (num_class + 1)
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
        num_classes = 1
     
    y_pred = K.reshape(y_pred, (-1, num_classes))#num_class
    y_true = K.reshape(y_true, (-1, num_classes))#num_class
    
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    flag = tf.convert_to_tensor(-1, dtype='float64')
    for i in range(num_classes-1):
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        cond = (K.sum(union) > 0) & (K.sum(tf.to_int32(true_labels)) > 0)
        res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
        iou.append(res)
    iou = tf.stack(iou)
    legal_labels = tf.greater(iou, flag)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)