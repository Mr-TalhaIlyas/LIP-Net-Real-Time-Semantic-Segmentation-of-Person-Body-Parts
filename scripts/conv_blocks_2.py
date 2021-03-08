import tensorflow as tf
import numpy as np
from layers import MinPooling2D
from conv_blocks_1 import  SE_ResNet, SE_block, SE_ResNet0
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    import keras.backend as K
    from keras.regularizers import l2, l1
    #from tensorflow.keras.layers.merge import concatenate
    from keras.models import Model
    from keras.layers import Input, BatchNormalization, Activation, ZeroPadding2D, Reshape, Lambda
    from keras.layers import GlobalAveragePooling2D, Dense, Permute, multiply, add, PReLU, concatenate
    from keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D, UpSampling2D
    from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
    from keras.backend import resize_images, int_shape
if tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
    from tensorflow.keras.regularizers import l2, l1
    #from tensorflow.keras.layers.merge import concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, ZeroPadding2D, Reshape, Lambda
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Permute, multiply, add, PReLU, concatenate
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D, UpSampling2D
    from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
    from tensorflow.keras.backend import resize_images, int_shape 
    
tf.compat.v1.get_default_graph
#%%
def WC(input_tensor, n_filters, kernel_size):

    # Wide Conv
    x1 = Conv2D(n_filters, kernel_size = (kernel_size,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x1 = Conv2D(n_filters, kernel_size = (1,kernel_size), kernel_initializer = 'he_normal', padding = 'same')(x1)
    

    x2 = Conv2D(n_filters, kernel_size = (1,kernel_size), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x2 = Conv2D(n_filters, kernel_size = (kernel_size,1), kernel_initializer = 'he_normal', padding = 'same')(x2)    
    
    xa = add([x1,x2])
    return xa

def ES(input_tensor, n_filters):
    # Wide Conv
    x1 = Conv2D(n_filters, kernel_size = (15,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x1 = Conv2D(n_filters, kernel_size = (1,15), kernel_initializer = 'he_normal', padding = 'same')(x1)
 
    x2 = Conv2D(n_filters, kernel_size = (13,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x2 = Conv2D(n_filters, kernel_size = (1,13), kernel_initializer = 'he_normal', padding = 'same')(x2)
    
    x3 = Conv2D(n_filters, kernel_size = (11,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x3 = Conv2D(n_filters, kernel_size = (1,11), kernel_initializer = 'he_normal', padding = 'same')(x3)
    
    x4 = Conv2D(n_filters, kernel_size = (9,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x4 = Conv2D(n_filters, kernel_size = (1,9), kernel_initializer = 'he_normal', padding = 'same')(x4)
    
    xadd = add([x1,x2, x3, x4])
    xadd = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(xadd)
    xadd = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(xadd)

    xskip = Conv2D(n_filters, kernel_size = (1,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    
    x_op = concatenate([xskip, xadd])
    return x_op

def PP(input_tensor, n_filters, level):
    feature_map_shape = int_shape(input_tensor)[1:3]
    pool_sizes = [int(np.round(feature_map_shape[0] / level)), int(np.round(feature_map_shape[1] / level))]
    
    x = AveragePooling2D(pool_size=pool_sizes, strides=pool_sizes, padding='same')(input_tensor)
    
    x = Conv2D(n_filters//4, kernel_size = (1,1), kernel_initializer = 'he_normal', padding = 'same')(x)
    
    x = UpSampling2D(size=pool_sizes, data_format=None, interpolation='bilinear')(x)
    
    return x

def PSP_module(input_tensor, n_filters):
    
    x1 = PP(input_tensor, n_filters, 1)
    x2 = PP(input_tensor, n_filters, 2)
    x3 = PP(input_tensor, n_filters, 4)
    x4 = PP(input_tensor, n_filters, 8)
    
    xc = concatenate([input_tensor, x1, x2, x3, x4])
    return xc

def ASPP_v2(input_tensor, n_filters):
    r_filters = n_filters//4
    
    x1 = Conv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 6, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x1)
    x1 = BatchNormalization()(x1)
    
    x8 = Conv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 12, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x8 = BatchNormalization()(x8)
    x8 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x8)
    x8 = BatchNormalization()(x8)
    x8 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x8)
    x8 = BatchNormalization()(x8)
    
    x16 = Conv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 18, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x16 = BatchNormalization()(x16)
    x16 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x16)
    x16 = BatchNormalization()(x16)
    x16 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x16)
    x16 = BatchNormalization()(x16)
    
    x24 = Conv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 24, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x24 = BatchNormalization()(x24)
    x24 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x24)
    x24 = BatchNormalization()(x24)
    x24 = Conv2D(filters = r_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(x24)
    x24 = BatchNormalization()(x24)
    
    c = add([x1, x8, x16, x24])

    return c

def ASPP_v3(input_tensor, n_filters, img, downsample_by):
    
    x1 = SeparableConv2D(filters = n_filters, kernel_size = (1, 1), depthwise_initializer='he_normal', pointwise_initializer='he_normal',  padding = 'same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x8 = SeparableConv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 6, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x8 = BatchNormalization()(x8)
    x16 = SeparableConv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 12, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x16 = BatchNormalization()(x16)
    x24 = SeparableConv2D(filters = n_filters, kernel_size = (3, 3), dilation_rate = 18, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x24 = BatchNormalization()(x24)
    
    img = MaxPooling2D(pool_size=downsample_by, strides=downsample_by, padding='same')(img)
    c = concatenate([x1, x8, x16, x24, img])
    xc = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal',  padding = 'same')(c)

    return xc
def GCN(input_tensor, n_filters, kernel_size):
    
    x1 = Conv2D(n_filters, kernel_size = (kernel_size,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x1 = Conv2D(n_filters, kernel_size = (1,kernel_size), kernel_initializer = 'he_normal', padding = 'same')(x1)
    

    x2 = Conv2D(n_filters, kernel_size = (1,kernel_size), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x2 = Conv2D(n_filters, kernel_size = (kernel_size,1), kernel_initializer = 'he_normal', padding = 'same')(x2)    
    
    xa = add([x1,x2])
    return xa
def BR(input_tensor, n_filters):
    
    x = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(x)
    
    x_add = add([x, input_tensor]) 
    return x_add

def PAM(input_tensor):
    
    A = input_tensor  #  WxHxC => NXC
    A_shape = int_shape(A)[1:4]
    N = int(A_shape[0] * A_shape[1])
    n_filters = A_shape[2]
    
    B = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(A)
    B = Reshape((N, n_filters))(B) #(N,C)
    
    C = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(A)
    C = Reshape((N, n_filters))(C) #(N,C)
    
    mul1 = tf.linalg.matmul(C, B, transpose_b=True)#(N,C)*(C,N)=(N,N)
    mul1 = Activation('softmax')(mul1)
    
    D = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(A)
    D = Reshape((N, n_filters))(D)#(N,C)
    
    mul2 = tf.linalg.matmul(mul1, D)#(N,N)*(N,C)=(N,C)
    mul2 = Reshape((A_shape[0] , A_shape[1], n_filters))(mul2)
        
    E = add([mul2, A])
   
    return E

def CAM(input_tensor):
    
    A = input_tensor
    A_shape = int_shape(A)[1:4]
    N = int(A_shape[0] * A_shape[1])
    n_filters = A_shape[2]
    
    Ar = Reshape((N, n_filters))(A)#(N,C)
    
    mul1 = tf.linalg.matmul(Ar, Ar, transpose_a=True)#(C,N)*(N,C)=(C,C)
    mul1 = Activation('softmax')(mul1)
    
    mul2 = tf.linalg.matmul(Ar, mul1)#(N,C)*(C,C)=(N,C)
    mul2 = Reshape((A_shape[0] , A_shape[1], n_filters))(mul2)
    
    E = add([mul2, A])
    
    return E

def BAM_MLP(input_tensor, filters, ratio = 8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input_tensor._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, kernel_initializer='he_normal')(se)
    se = Dense(filters, kernel_initializer='he_normal')(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    return se

def BAM(input_tensor, n_filters, kernel_size, dil_rate, batchnorm = True):
    filters_r = int(n_filters/4)
    # Channel attention
    x_c = BAM_MLP(input_tensor, n_filters, 8)
    # Spatioal attention
    x_s = Conv2D(filters = filters_r, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s)
    # 2nd conv layer
    #x_s = SeparableConv2D(filters = filters_r, kernel_size = (kernel_size, kernel_size), dilation_rate = dil_rate, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(x_s)
    x_s = Conv2D(filters = filters_r, kernel_size = (kernel_size, kernel_size), dilation_rate = dil_rate, kernel_initializer = 'he_normal', padding = 'same')(x_s)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s)
    x_s = Conv2D(filters = filters_r, kernel_size = (kernel_size, kernel_size), dilation_rate = dil_rate, kernel_initializer = 'he_normal', padding = 'same')(x_s)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s)
    # 3rd conv layer
    x_s = Conv2D(filters = n_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal',  padding = 'same')(x_s)
    x_s = BatchNormalization()(x_s)
    
    x1 = add([x_c, x_s])
    x1 = Activation('sigmoid')(x1)
    
    x2 = multiply([input_tensor, x1])
    x_op = add([input_tensor, x2])
    return x_op

def CBAM_MLP(input_tensor, filters, ratio = 8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input_tensor._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se1 = GlobalAveragePooling2D()(input_tensor)
    se1 = Reshape(se_shape)(se1)
    
    se2 = GlobalMaxPooling2D()(input_tensor)
    se2 = Reshape(se_shape)(se2)
    
    fc1 = Dense(filters // ratio, kernel_initializer='he_normal')
    fc2 = Dense(filters, kernel_initializer='he_normal')
    
    se1 = fc1(se1)
    se1 = fc2(se1)
    se2 = fc1(se2)
    se2 = fc2(se2)

    if K.image_data_format() == 'channels_first':
        se1 = Permute((3, 1, 2))(se1)
        se2 = Permute((3, 1, 2))(se2)

    se_add = add([se1, se2])
    se_add = Activation('sigmoid')(se_add)
    x = multiply([input_tensor, se_add])
    return x

def CBAM(input_tensor, n_filters):
    # Channel attention DepthMaxPool, DepthAvgPool
    x_ca = CBAM_MLP(input_tensor, n_filters, 8)
    # Spatioal attention
    x_sa = Conv2D(filters = 1, kernel_size = (1, 1), kernel_initializer = 'he_normal',  padding = 'same')(x_ca)
    x_sa = Conv2D(1, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(x_sa)
    x_sa = BatchNormalization()(x_sa)
    x_sa = Activation('sigmoid')(x_sa)
    
    x = multiply([input_tensor, x_sa])
    x = add([x, input_tensor])
    return x


def Red_net_ip(img):
    #b, w, h, _ = img.shape
    #empty = tf.keras.backend.zeros(shape= (b, w, h, 1), dtype=img.dtype)
    grays = tf.image.rgb_to_grayscale(img)
    ip = tf.concat([img, grays], axis=3)
    return ip















































