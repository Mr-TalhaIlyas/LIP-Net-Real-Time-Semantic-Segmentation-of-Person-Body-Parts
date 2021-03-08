'''
Bi-Isame-Allah
'''
import tensorflow as tf
import numpy as np
from conv_blocks_1 import  SE_ResNet, conv2d_block, SE_ResNet0, SE_block
from models import use_customdropout, num_of_classes
if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout, SpatialDropout2D, PReLU, Lambda
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout, ZeroPadding2D, Reshape, Lambda
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Permute, multiply, add
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D, UpSampling2D
    from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
    from tensorflow.keras.backend import int_shape
    
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    from keras.models import Model
    from keras.layers import Input, BatchNormalization, Activation, Dropout, SpatialDropout2D, PReLU, Lambda
    from keras.layers import concatenate
    from keras.regularizers import l2, l1
    from keras.models import Model
    from keras.layers import Input, BatchNormalization, Activation, Dropout, ZeroPadding2D, Reshape, Lambda
    from keras.layers import GlobalAveragePooling2D, Dense, Permute, multiply, add
    from keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D, UpSampling2D
    from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
    from keras.backend import int_shape
    
use_mydropout = use_customdropout()

if use_mydropout == True:
        from layers import Dropout
elif use_mydropout == False:
    if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.0':
        from keras.layers import Dropout
    if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0':
        from tensorflow.keras.layers import Dropout
        
num_class = num_of_classes()

if num_class ==2:
    output_ch = 1
else:
    output_ch = num_class
    
#%%
    
def APP0(input_tensor, pool_window):
    
    avg = AveragePooling2D(pool_size=(pool_window,pool_window))(input_tensor)
    maax = MaxPooling2D(pool_size=(pool_window,pool_window))(input_tensor)
    x_op = add([avg, maax])
    
    return x_op

def DAM(l_f_maps, h_f_maps, n_filters, g_kernal, image, pool_w, activation):
    #filters_r = int(n_filters/4)
    
    x_ip = Conv2D(48, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(l_f_maps)
    
    img = APP0(image, pool_window = pool_w)
    x_ip = concatenate([x_ip, img])
    # Global Conv
    x1 = Conv2D(n_filters, kernel_size = (g_kernal,1), kernel_initializer = 'he_normal', padding = 'same')(x_ip)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)
    x2 = Conv2D(n_filters, kernel_size = (1,g_kernal), kernel_initializer = 'he_normal', padding = 'same')(x_ip)
    x2 = BatchNormalization()(x2)
    x2 = Activation(activation)(x2)
    
    xd0 = add([x1,x2])

    x3 = DepthwiseConv2D(kernel_size = (3, 3), depthwise_initializer='he_normal', padding = 'same')(xd0)
    x3 = BatchNormalization()(x3)
    x3 = Activation(activation)(x3)
    x4 = DepthwiseConv2D(kernel_size = (3, 3), depthwise_initializer='he_normal', padding = 'same')(xd0)   
    x4 = BatchNormalization()(x4)
    x4 = Activation(activation)(x4)
       
    avg = AveragePooling2D()(x3)
    avg = SE_block(avg, n_filters, ratio = 4)#2
    maax = MaxPooling2D()(x4)
    maax = SE_block(maax, n_filters, ratio = 4)#2
    
    x_sum = add([avg,maax])
    x_sig = Activation(activation)(x_sum)

    
    
    x_c = concatenate([x_sig,h_f_maps])#,img
    
    #x_op = Conv2D(filters = n_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal',  padding = 'same')(x_c)
    x_op = SE_ResNet(x_c, n_filters, kernel_size = 3, batchnorm = True, dil_rate = 1)

    return x_op

def PDC(input_tensor, n_filters, dropout, activation='relu'):

    x1 = SeparableConv2D(filters = 512, kernel_size = (1, 1), depthwise_initializer='he_normal', pointwise_initializer='he_normal',  padding = 'same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x8 = SeparableConv2D(filters = 512, kernel_size = (3, 3), dilation_rate = 6, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x8 = BatchNormalization()(x8)
    x16 = SeparableConv2D(filters = 512, kernel_size = (3, 3), dilation_rate = 12, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x16 = BatchNormalization()(x16)
    x24 = SeparableConv2D(filters = 512, kernel_size = (3, 3), dilation_rate = 18, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x24 = BatchNormalization()(x24)
        
    c = concatenate([x1, x8, x16, x24])
    c = Conv2D(filters = n_filters, kernel_size = (1,1), kernel_initializer = 'he_normal', padding = 'same')(c)
    c = BatchNormalization()(c)
    c = Activation(activation)(c)
    c = Dropout(dropout)(c)
    return c

def ASPP_v3(input_tensor, img, dropout, downsample_by, activation='relu'):
    
    x1 = SeparableConv2D(filters = 256, kernel_size = (1, 1), depthwise_initializer='he_normal', pointwise_initializer='he_normal',  padding = 'same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x8 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 6, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x8 = BatchNormalization()(x8)
    x16 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 12, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x16 = BatchNormalization()(x16)
    x24 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 18, depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(input_tensor)
    x24 = BatchNormalization()(x24)
    
    img = AveragePooling2D(pool_size=downsample_by, strides=downsample_by, padding='same')(img)
    
    c = concatenate([x1, x8, x16, x24, img])
    c = Conv2D(filters = 256, kernel_size = (1,1), kernel_initializer = 'he_normal', padding = 'same')(c)
    c = BatchNormalization()(c)
    c = Activation(activation)(c)
    c = Dropout(dropout)(c)

    return c

def Xception(input_img, n_filters, dropout, activation = 'relu', m_flow = 16, batchnorm = True):
    
    # Entry FLow
    #input
    ip = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(input_img)
    ip = BatchNormalization()(ip)
    ip = Activation(activation)(ip)
    ip = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal',padding = 'same')(ip)
    ip = BatchNormalization()(ip)
    ip = Activation(activation)(ip)       # *******___1/2 times smaller than ip___********
    # 1st Residual connection
    res1 = Conv2D(filters = 128, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(ip)
    res1 = BatchNormalization()(res1)
    # Block 1
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(ip)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)
    b1 = Dropout(dropout)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)
    b1 = Dropout(dropout)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)       # *******___1/4 times smaller than ip___********
    b1 = Dropout(dropout)(b1)
    b1 = add([b1, res1])
    # 2nd Residual connection
    res2 = Conv2D(filters = 256, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2))(b1)
    res2 = BatchNormalization()(res2)
    # Block 2
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)
    b2 = Dropout(dropout)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)
    b2 = Dropout(dropout)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)       # *******___1/8 times smaller than ip___********
    b2 = Dropout(dropout)(b2)
    b2 = add([b2, res2])
    # 3rd Residual connection
    res3 = Conv2D(filters = 768, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(b2)
    res3 = BatchNormalization()(res3)
    # Block 3
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)
    b3 = Dropout(dropout)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)
    b3 = Dropout(dropout)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)       # *******___1/16 times smaller than ip___********
    b3 = Dropout(dropout)(b3)
    b3 = add([b3, res3])
    # Middle Flow
    # 4th residual connection  8
    res4 = b3
    for i in range(m_flow):
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = add([b4, res4])
        res4 = b4
    
    # Exit Flow
    
    # 5th residual connection
    res5 = Conv2D(filters = 1024, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (1,1), padding = 'same')(b4)
    res5 = BatchNormalization()(res5)
    # Block 5  2
    b5 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)
    b5 = Dropout(dropout)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)
    b5 = Dropout(dropout)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), strides = (1,1), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)       # *******___1/32 times smaller than ip___********
    b5 = Dropout(dropout)(b5)
    b5 = add([b5, res5])

    # Block 6
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    b6 = SeparableConv2D(filters = 2048, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    
    
    #*************************************************************************
    # Encoder to Decoder
    #*************************************************************************
    #Transition
    ctr = ASPP_v3(b6, input_img, dropout, downsample_by = 16)
    
    # Upsampling
    up = UpSampling2D(size=((4,4)), interpolation='bilinear')(ctr)#x4 times upsample 
    
    up1 = Conv2D(48, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(b1)
    up1 = BatchNormalization()(up1)
    up1 = Activation(activation)(up1)
    upc = concatenate([up1, up])
    
    up2 = Conv2D(256, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc)
    up2 = BatchNormalization()(up2)
    up2 = Activation(activation)(up2)
    up2 = UpSampling2D(size=((4,4)), interpolation='bilinear')(up2)#x4 times upsample 
    
    outputs = Conv2D(output_ch, (1, 1), kernel_initializer = 'he_normal')(up2)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    
    return model

def Xception_v2(input_img, n_filters, dropout, activation = 'relu', m_flow = 16, batchnorm = True):#5e-4

    # Entry FLow
    #input
    ip = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer = 'he_normal',  strides = (2,2), padding = 'same')(input_img)
    ip = BatchNormalization()(ip)
    ip = Activation(activation)(ip)
    ip = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(ip)
    ip = BatchNormalization()(ip)
    ip = Activation(activation)(ip)       # *******___1/2 times smaller than ip___********
    # 1st Residual connection
    res1 = Conv2D(filters = 128, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(ip)
    res1 = BatchNormalization()(res1)
    # Block 1
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(ip)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)
    b1 = Dropout(dropout)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)
    b1 = Dropout(dropout)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)       # *******___1/4 times smaller than ip___********
    b1 = Dropout(dropout)(b1)
    b1 = add([b1, res1])
    # 2nd Residual connection
    res2 = Conv2D(filters = 256, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2))(b1)
    res2 = BatchNormalization()(res2)
    # Block 2
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)
    b2 = Dropout(dropout)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)
    b2 = Dropout(dropout)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)       # *******___1/8 times smaller than ip___********
    b2 = Dropout(dropout)(b2)
    b2 = add([b2, res2])
    # 3rd Residual connection
    res3 = Conv2D(filters = 768, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(b2)
    res3 = BatchNormalization()(res3)
    # Block 3
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)
    b3 = Dropout(dropout)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)
    b3 = Dropout(dropout)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)       # *******___1/16 times smaller than ip___********
    b3 = Dropout(dropout)(b3)
    b3 = add([b3, res3])
    # Middle Flow
    # 4th residual connection  8
    res4 = b3
    for i in range(m_flow):
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = add([b4, res4])
        res4 = b4
    
    # Exit Flow
    
    # 5th residual connection
    res5 = Conv2D(filters = 1024, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (1,1), padding = 'same')(b4)
    res5 = BatchNormalization()(res5)
    # Block 5  2
    b5 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)
    b5 = Dropout(dropout)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)
    b5 = Dropout(dropout)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), strides = (1,1), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)       # *******___1/32 times smaller than ip___********
    b5 = Dropout(dropout)(b5)
    b5 = add([b5, res5])

    # Block 6
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    b6 = SeparableConv2D(filters = 2048, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    
    
    #*************************************************************************
    # Encoder to Decoder
    #*************************************************************************
    #Transition
    ctr = ASPP_v3(b6, input_img, dropout, downsample_by = 16)
    
    # Upsampling
    up = UpSampling2D(size=((2,2)), interpolation='bilinear')(ctr)#x2 times upsample 
    
    up1 = Conv2D(48, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(b2)
    up1 = BatchNormalization()(up1)
    up1 = Activation(activation)(up1)
    
    upc = concatenate([up1, up])
    up2 = Conv2D(256, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc)
    up2 = BatchNormalization()(up2)
    up2 = Activation(activation)(up2)
    
    up2 = UpSampling2D(size=((2,2)), interpolation='bilinear')(up2)#x2 times upsample 
    
    up3 = Conv2D(48, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(b1)
    up3 = BatchNormalization()(up3)
    up3 = Activation(activation)(up3)
    
    upc2 = concatenate([up2, up3])
    upc2 = Conv2D(256, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc2)
    upc2 = BatchNormalization()(upc2)
    upc2 = Activation(activation)(upc2)
    
    upc2 = UpSampling2D(size=((2,2)), interpolation='bilinear')(upc2)#x2 times upsample 
    
    up4 = Conv2D(18, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ip)
    up4 = BatchNormalization()(up4)
    up4 = Activation(activation)(up4)
    
    upc3 = concatenate([upc2, up4])
    upc3 = Conv2D(256, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc3)
    upc3 = BatchNormalization()(upc3)
    upc3 = Activation(activation)(upc3)
    
    upc3 = UpSampling2D(size=((2,2)), interpolation='bilinear')(upc3)
    
    outputs = Conv2D(output_ch, (1, 1), kernel_initializer = 'he_normal')(upc3)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    
    return model

def Xception_seed(input_img, n_filters, dropout, activation = 'relu', m_flow = 16, batchnorm = True):
    
    # Entry FLow
    #input
    ip = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(input_img)
    ip = BatchNormalization()(ip)
    ip = Activation(activation)(ip)
    ip = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(ip)
    ip = BatchNormalization()(ip)
    ip = Activation(activation)(ip)       # *******___1/2 times smaller than ip___********
    # 1st Residual connection
    res1 = Conv2D(filters = 128, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(ip)
    res1 = BatchNormalization()(res1)
    # Block 1
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(ip)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)
    b1 = Dropout(dropout)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)
    b1t = Dropout(dropout)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1t)
    b1 = BatchNormalization()(b1)
    b1 = Activation(activation)(b1)       # *******___1/4 times smaller than ip___********
    b1 = Dropout(dropout)(b1)
    b1 = add([b1, res1])
    # 2nd Residual connection
    res2 = Conv2D(filters = 256, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2))(b1)
    res2 = BatchNormalization()(res2)
    # Block 2
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)
    b2 = Dropout(dropout)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)
    b2t = Dropout(dropout)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2t)
    b2 = BatchNormalization()(b2)
    b2 = Activation(activation)(b2)       # *******___1/8 times smaller than ip___********
    b2 = Dropout(dropout)(b2)
    b2 = add([b2, res2])
    # 3rd Residual connection
    res3 = Conv2D(filters = 768, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (2,2), padding = 'same')(b2)
    res3 = BatchNormalization()(res3)
    # Block 3
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)
    b3 = Dropout(dropout)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)
    b3t = Dropout(dropout)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), strides = (2,2), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3t)
    b3 = BatchNormalization()(b3)
    b3 = Activation(activation)(b3)       # *******___1/16 times smaller than ip___********
    b3 = Dropout(dropout)(b3)
    b3 = add([b3, res3])
    # Middle Flow
    # 4th residual connection  8
    res4 = b3
    for i in range(m_flow):
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = Activation(activation)(b4)
        b4 = Dropout(dropout)(b4)
        b4 = add([b4, res4])
        res4 = b4
    
    # Exit Flow
    
    # 5th residual connection
    res5 = Conv2D(filters = 1024, kernel_size = (1,1), kernel_initializer = 'he_normal', strides = (1,1), padding = 'same')(b4)
    res5 = BatchNormalization()(res5)
    # Block 5  2
    b5 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)
    b5 = Dropout(dropout)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)
    b5 = Dropout(dropout)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), strides = (1,1), depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = Activation(activation)(b5)       # *******___1/32 times smaller than ip___********
    b5 = Dropout(dropout)(b5)
    b5 = add([b5, res5])

    # Block 6
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)
    b6 = SeparableConv2D(filters = 2048, kernel_size = (3, 3), dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = Activation(activation)(b6)
    b6 = Dropout(dropout)(b6)    
    
    #*************************************************************************
    # Encoder to Decoder
    #*************************************************************************
    #Transition
    #connecting encoder decoder
    c6 = PDC(b6, 2048, dropout)
    #c6 = ASPP_v3(b6, input_img, dropout, downsample_by = 16)
    
    # Expanding Path
    dam6 = DAM(b3t, c6, 512, 15, input_img, 8, activation)
    u6 = UpSampling2D(interpolation='bilinear')(dam6) # 2x upsample

    dam7 = DAM(b2t, u6, 256, 15, input_img, 4, activation)   
    u7 = UpSampling2D(interpolation='bilinear')(dam7) # 2x upsample
    
    dam8 = DAM(b1t, u7, 128, 15, input_img, 2, activation)   
    u8 = UpSampling2D(interpolation='bilinear')(dam8) # 2x upsample
    
    up4 = Conv2D(18, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(ip)
    up4 = BatchNormalization()(up4)
    up4 = Activation(activation)(up4)
    
    u8 = concatenate([u8, up4])
    u8 = Conv2D(128, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(u8)
    u8 = BatchNormalization()(u8)
    u8 = Activation(activation)(u8)
    u9 = UpSampling2D(size=((2,2)), interpolation='bilinear')(u8) # 2x upsample
    
    outputs = Conv2D(output_ch, (1, 1), kernel_initializer = 'he_normal')(u9)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    
    return model















































