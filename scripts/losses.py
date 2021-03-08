import tensorflow as tf
import numpy as np
from models import  num_of_classes
if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
    from keras.layers.merge import concatenate
    from keras.layers import Activation
    import keras.backend as K
if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Activation, concatenate
    import tensorflow_addons as tfa

'''
In this on the fly lossses the ground truths are converted on they fly into the categorical type and only 
hybrid and tri brid losses are doing that if you wann use only one loss then convert them first
'''
#-------------------------------------------------------------Dice Loss Function-----------------------
num_class = num_of_classes()


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def Weighted_BCEnDice_loss(y_true, y_pred):
    
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
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss
#%%
def focal_CE(y_true, y_pred):
    
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
        
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    loss = tfa.losses.sigmoid_focal_crossentropy(y_true = y_true, y_pred = y_pred, 
                                                 alpha = 0.25, gamma  = 2.0, from_logits = True)+sigmoid_cross_entropy_balanced(y_pred, y_true) 
    return loss
#%%
def HED_loss(y_true, y_pred):
    
    y_true = y_true * (num_class + 1)  
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
        
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    loss = sigmoid_cross_entropy_balanced(y_pred, y_true) 
    return loss

def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):
    """
    From:

	https://github.com/moabitcoin/holy-edge/blob/master/hed/losses.py

    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0':
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
    if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
#--------------------------------------------------------------Focal Loss---------------------------------------------
gamma=2.
alpha=.25
epsilon = 1.e-9
def focal_loss(y_true, y_pred):
    
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_true = y_true * (num_class + 1)  
        y_true = K.squeeze(y_true, 3)
        y_true = tf.cast(y_true, "int32")
        y_true = tf.one_hot(y_true, num_class, axis=-1)
        if num_class==2:
            y_true = y_true[:,:,:,0:1]

        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        loss = tf.reduce_mean(reduced_fl) #+ multi_iou_loss(y_true, y_pred, smooth=1)
        return loss
#---------------------------------------------------multi_iou_loss----------------------------------------

def multi_iou_loss(y_true, y_pred, smooth=1):
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    loss = 1-iou
    return loss

#%%
def focal_loss2(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true,1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. -1e-4)) * tf.pow(1-hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1-hm_pred, 1e-4, 1. -1e-4)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda :(pos_loss + neg_loss)/num_pos, lambda :neg_loss)
    return cls_loss


def soft_dice_loss(y_true, y_pred, epsilon=1e-7): 
    ''' 
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean(numerator / (denominator + epsilon)) # average over classes and batch


from keras.losses import binary_crossentropy

def built_loss(y_true, y_pred):
    
    y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    y_true = y_true * (num_class + 1)  
    y_true = K.squeeze(y_true, 3)
    y_true = tf.cast(y_true, "int32")
    y_true = tf.one_hot(y_true, num_class, axis=-1)
    if num_class==2:
        y_true = y_true[:,:,:,0:1]
        
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    
    loss1 = binary_crossentropy(y_true, y_pred)
    #loss2 = soft_dice_loss(y_true, y_pred, epsilon=1e-6)
    loss = loss1# + loss2
    return loss












    
    