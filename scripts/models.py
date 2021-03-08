'''
Bi-Isame-Allah
This script has following networks
0. Seed Net version 1 & 101
1. Seg_net & Seg_net_original (in these n_filters=32)
2. U_net
3. U_net WC
4. ES_net
5. FCN_8s
6. PSP_net (op = 1/8 x ip)
7. Deeplab_v3
8. GCN (ip = 512x512)
9. DAN (op = 1/8 x ip)
'''
import tensorflow as tf
from conv_blocks_1 import  SE_ResNet, conv2d_block, SE_ResNet0
from conv_blocks_1 import  avg_img_pyramid, max_img_pyramid, dense_skip, dense_skip0, global_dil, PDC, ResNet_block_op, DAM
from conv_blocks_2 import ASPP_v2, ASPP_v3, PSP_module, Red_net_ip, WC, ES, PSP_module, ASPP_v3, ASPP_v2, GCN, BR, PAM, CAM, BAM, CBAM
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D, MinPooling2D
if tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0' or tf.__version__ == '2.2.0-rc2':
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
    from tensorflow.keras.layers import MaxPooling2D, concatenate
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
    import tensorflow.keras.backend as K
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    from keras.models import Model
    from keras.regularizers import l2, l1
    from keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
    from keras.layers import MaxPooling2D, concatenate
    from keras.optimizers import Adam, Nadam, SGD
    import keras.backend as K

def use_customdropout():
    use_mydropout = True     # 1 for Ture
    return use_mydropout  # 0 for False

use_mydropout = use_customdropout()
if use_mydropout == True:
        from layers import Dropout
elif use_mydropout == False:
    if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
        from keras.layers import Dropout
    if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0':
        from tensorflow.keras.layers import Dropout

'''
For binary segmentation set num_class=2
'''
def num_of_classes():
    num_class = 15
    return num_class

num_class = num_of_classes()
if num_class == 2:
    output_ch = 1
else:
    output_ch = num_class
    
starting_ch = 16 # For networks other than Seed Net
#%%**********************************************SEED_Netv1**********************************************
def SEED_Netv1(input_img, n_filters, dropout, batchnorm = True, activation = 'relu'):
    """Function to define the UNET Model"""
    #Making image pyramids for concatinaing at later stages will simply resize the images
    avg_pyramid1, avg_pyramid2, avg_pyramid3, avg_pyramid4 = avg_img_pyramid(input_img) #via average pooling
    max_pyramid1, max_pyramid2, max_pyramid3, max_pyramid4 = max_img_pyramid(input_img) #via max pooling
    
    # Contracting Path
    c1 = SE_ResNet(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)

    #connecting encoder decoder
    #c5 = parallel_dil(c5, n_filters * 16, kernel_size = 3) #with parallet dil
    c5 = dense_skip(c5, avg_pyramid4, max_pyramid4, n_filters * 16, kernel_size = 7)#with dense skip
    #c5 = PDC(c5, n_filters * 16)

    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    ds6 = dense_skip(c4, avg_pyramid3, max_pyramid3, n_filters * 8, kernel_size = 9)
    u6 = concatenate([u6, ds6])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    ds7 = dense_skip(c3, avg_pyramid2, max_pyramid2, n_filters * 4, kernel_size = 11)
    u7 = concatenate([u7, ds7])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    ds8 = dense_skip(c2, avg_pyramid1, max_pyramid1, n_filters * 2, kernel_size = 13)
    u8 = concatenate([u8, ds8])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    ds9 = dense_skip0(c1, input_img, n_filters * 1, kernel_size = 15)
    u9 = concatenate([u9, ds9])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    
    c_out = ResNet_block_op(c9, output_ch, kernel_size = 3, batchnorm = batchnorm)
    outputs = Conv2D(output_ch, (1, 1))(c_out)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%**********************************************SEED_Netv92**********************************************
def SEED_Netv101(input_img, n_filters, dropout, weight_decay=False, batchnorm = True, activation = 'relu'):
    """Function to define the UNET Model"""
    if weight_decay == True:
        weight_decay = l2(5e-4)
    else:
        weight_decay = None
    c0 = Conv2D(16, kernel_size = (7, 7), strides=(2, 2), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation(activation)(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2, activation = activation)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2, activation = activation)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4, activation = activation)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4, activation = activation)
    #p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(c4)
    
    c5 = SE_ResNet(p4, n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8, activation = activation)
    c5 = SE_ResNet0(c5, n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8, activation = activation)
    #p4 = MaxPooling2D((2, 2))(c4)
    p5 = Dropout(dropout)(c5)
    
    c6 = SE_ResNet(p5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)
    c6 = SE_ResNet0(c6, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1, activation = activation)

    #connecting encoder decoder
    c6 = PDC(c6, n_filters * 16)
    #c6 = PSP_module(c6, n_filters * 16)
    #c6 = ASPP_v3(c6, n_filters*16, input_img, downsample_by = 16)
    #c6 = ASPP_v2(c6, n_filters*16)
    
    # Expanding Path
    dam6 = DAM(c3, c6, n_filters*16, 15, input_img, 8)
    u6 = UpSampling2D(interpolation='bilinear')(dam6)

    dam7 = DAM(c2, u6, n_filters*8, 15, input_img, 4)   
    u7 = UpSampling2D(interpolation='bilinear')(dam7)
    
    dam8 = DAM(c1, u7, n_filters*4, 15, input_img, 2)   
    u8 = UpSampling2D(size=(4,4), interpolation='bilinear')(dam8)
    
    outputs = Conv2D(output_ch, (1, 1), kernel_initializer = 'he_normal')(u8)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%%
def FCN_8s(input_img, n_filters, dropout, kernel=3, batchnorm = True): 
     # Block 1
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)
    
    # up convolutions and sum
    p5u = UpSampling2D(size = (2,2),interpolation='bilinear')(p5)
    p5u = Conv2D(n_filters * 8, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p5u)
    sum45 = add([p5u, p4])  
    sum45 = UpSampling2D(size = (2,2),interpolation='bilinear')(sum45)
    sum45 = Conv2D(n_filters * 4, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(sum45)
    sum453 = add([sum45, p3])   
    sum453 = UpSampling2D(size = (2,2),interpolation='bilinear')(sum453) 
    sum453 = Conv2D(n_filters * 2, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(sum453)
    sum4532 = add([sum453, p2])   
    sum4532 = UpSampling2D(size = (2,2),interpolation='bilinear')(sum4532) 
    sum4532 = Conv2D(n_filters * 1, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(sum4532)
    sum45321 = add([sum4532, p1])   
    sum45321 = UpSampling2D(size = (2,2),interpolation='bilinear')(sum45321) 
    
    outputs = Conv2D(output_ch, 1)(sum45321)#, activation = 'softmax'
    
    model = Model(inputs=[input_img], outputs=[outputs])
    
    return model
#%%
def Seg_net(input_img, n_filters, dropout, kernel=3, batchnorm = True):
    
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # encoder
    conv_1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_2 = SE_ResNet0(conv_1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  

    pool_1, mask_1 = MaxPoolingWithArgmax2D((2,2))(conv_2)

    conv_3 = SE_ResNet(pool_1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_4 = SE_ResNet0(conv_3, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    pool_2, mask_2 = MaxPoolingWithArgmax2D((2,2))(conv_4)

    conv_5 = SE_ResNet(pool_2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_6 = SE_ResNet0(conv_5, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_7 = SE_ResNet0(conv_6, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    pool_3, mask_3 = MaxPoolingWithArgmax2D((2,2))(conv_7)

    conv_8 = SE_ResNet(pool_3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_9 = SE_ResNet0(conv_8, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_10 = SE_ResNet0(conv_9, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    pool_4, mask_4 = MaxPoolingWithArgmax2D((2,2))(conv_10)

    conv_11 = SE_ResNet(pool_4, n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)#8
    conv_12 = SE_ResNet0(conv_11, n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)#8
    conv_13 = SE_ResNet0(conv_12, n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)#8

    pool_5, mask_5 = MaxPoolingWithArgmax2D((2,2))(conv_13)
    # decoder

    unpool_1 = MaxUnpooling2D((2,2))([pool_5, mask_5])

    conv_14 = SE_ResNet(unpool_1, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_15 = SE_ResNet0(conv_14, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_16 = SE_ResNet0(conv_15, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    unpool_2 = MaxUnpooling2D((2,2))([conv_16, mask_4])

    conv_17 = SE_ResNet(unpool_2, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_18 = SE_ResNet0(conv_17, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_19 = SE_ResNet(conv_18, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    unpool_3 = MaxUnpooling2D((2,2))([conv_19, mask_3])

    conv_20 = SE_ResNet(unpool_3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_21 = SE_ResNet0(conv_20, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_22 = SE_ResNet(conv_21, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    unpool_4 = MaxUnpooling2D((2,2))([conv_22, mask_2])

    conv_23 = SE_ResNet(unpool_4, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    conv_24 = SE_ResNet(conv_23, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    unpool_5 = MaxUnpooling2D((2,2))([conv_24, mask_1])

    conv_25 = SE_ResNet(unpool_5, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    conv_26 = Conv2D(output_ch, (1, 1), padding="valid")(conv_25)
    outputs = BatchNormalization()(conv_26)
    #outputs = Activation('softmax')(outputs)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model


#%%
def U_net(input_img, n_filters, dropout, batchnorm = True):

    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)

    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    outputs = Conv2D(output_ch, (1, 1))(c9)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%%
def Unet_WC(input_img, n_filters, dropout, batchnorm = True):
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #Transition
    ctr = WC(c5, n_filters*16, kernel_size=13)
    
    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(ctr)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    outputs = Conv2D(output_ch, (1, 1))(c9)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def ES_net(input_img, n_filters, dropout, batchnorm = True):
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #Transition
    ctr = WC(c5, n_filters*16, kernel_size=13)
    
    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(ctr)
    c4_es = ES(c4, n_filters * 8)
    u6 = concatenate([u6, c4_es])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    c3_es = ES(c3, n_filters * 4)
    u7 = concatenate([u7, c3_es])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    c2_es = ES(c2, n_filters * 2)
    u8 = concatenate([u8, c2_es])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    c1_es = ES(c1, n_filters * 1)
    u9 = concatenate([u9, c1_es])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    outputs = Conv2D(output_ch, (1, 1))(c9)#, activation='softmax
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def Pyramid_Unet(input_img, n_filters, dropout, batchnorm = True):
    """Function to define the UNET Model"""
    #Making image pyramids for concatinaing at later stages will simply resize the images
    avg_pyramid1, avg_pyramid2, avg_pyramid3, avg_pyramid4 = avg_img_pyramid(input_img) #via average pooling
    max_pyramid1, max_pyramid2, max_pyramid3, max_pyramid4 = max_img_pyramid(input_img) #via max pooling
    
    # Contracting Path
    c1 = SE_ResNet(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #connecting encoder decoder
    #c5 = parallel_dil(c5, n_filters * 16, kernel_size = 3) #with parallet dil
    c5 = dense_skip(c5, avg_pyramid4, max_pyramid4, n_filters * 16, kernel_size = 7)#with dense skip

    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    ds6 = dense_skip(c4, avg_pyramid3, max_pyramid3, n_filters * 8, kernel_size = 9)
    u6 = concatenate([u6, ds6])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 1, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    ds7 = dense_skip(c3, avg_pyramid2, max_pyramid2, n_filters * 4, kernel_size = 11)
    u7 = concatenate([u7, ds7])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 1, batchnorm = batchnorm, dil_rate = 1)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    ds8 = dense_skip(c2, avg_pyramid1, max_pyramid1, n_filters * 2, kernel_size = 13)
    u8 = concatenate([u8, ds8])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 1, batchnorm = batchnorm, dil_rate = 1)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    ds9 = dense_skip0(c1, input_img, n_filters * 1, kernel_size = 15)
    u9 = concatenate([u9, ds9])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 1, batchnorm = batchnorm, dil_rate = 1)
    
    outputs = Conv2D(output_ch, 1)(c9)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def PSP_net(input_img, n_filters, dropout, batchnorm = True):
    '''
    For this model output is 1/8 of the input size
    '''
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    #c4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(c4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #Transition
    ctr = PSP_module(c5, n_filters*16)
    
    outputs = Conv2D(output_ch, (1, 1))(ctr)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def Deeplab_v2(input_img, n_filters, dropout, batchnorm = True):
    
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    #p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(c4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #Transition
    ctr = ASPP_v2(c5, n_filters*16)
    
    up = Conv2D(output_ch, (1, 1), kernel_initializer = 'he_normal', activation='relu')(ctr)
    up = UpSampling2D(size=((8,8)), interpolation='bilinear')(up)#x8 times upsample directly
    
    outputs = Conv2D(output_ch, (1, 1))(up)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def Deeplab_v3(input_img, n_filters, dropout, batchnorm = True):
    
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #Transition
    ctr = ASPP_v3(c5, n_filters*16, input_img, downsample_by = 16)
    
    # Upsampling
    up = Conv2D(n_filters*8, (1, 1), kernel_initializer = 'he_normal', activation='relu')(ctr)
    up = UpSampling2D(size=((4,4)), interpolation='bilinear')(up)#x4 times upsample 
    
    up1 = Conv2D(n_filters, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c3)
    upc = concatenate([up1, up])
    
    up2 = Conv2D(n_filters*4, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(upc)
    up2 = UpSampling2D(size=((4,4)), interpolation='bilinear')(up2)#x4 times upsample 
    
    outputs = Conv2D(output_ch, (1, 1))(up2)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def GCN_net(input_img, n_filters, dropout, batchnorm = True):
    '''
    This one is designed for 512x512 input
    '''
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(dropout)(p5)
    
    # Expanding Path
    u5 = GCN(p5, output_ch, kernel_size = 15)
    u5 = BR(u5, output_ch)
    c6 = Conv2DTranspose(output_ch, (3, 3), strides = (2, 2), padding = 'same')(u5)
    c6 = Dropout(dropout)(c6)
    
    u4 = GCN(p4, output_ch, kernel_size = 15)
    u4 = BR(u4, output_ch)
    u4 = add([u4, c6])
    u4 = BR(u4, output_ch)
    c7 = Conv2DTranspose(output_ch, (3, 3), strides = (2, 2), padding = 'same')(u4)
    
    u3 = GCN(p3, output_ch, kernel_size = 15)
    u3 = BR(u3, output_ch)
    u3 = add([u3, c7])
    u3 = BR(u3, output_ch)
    c8 = Conv2DTranspose(output_ch, (3, 3), strides = (2, 2), padding = 'same')(u3)
    
    u2 = GCN(p2, output_ch, kernel_size = 15)
    u2 = BR(u2, output_ch)
    u2 = add([u2, c8])
    u2 = BR(u2, output_ch)
    c9 = Conv2DTranspose(output_ch, (3, 3), strides = (2, 2), padding = 'same')(u2)
    
    c10 = BR(c9, output_ch)
    c11 = Conv2DTranspose(output_ch, (3, 3), strides = (2, 2), padding = 'same')(c10)
    c11 = BR(c11, output_ch)
    
    outputs = Conv2D(output_ch, (1, 1))(c11)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def DAN_net(input_img, n_filters, dropout, batchnorm = True):
    '''
    For this model output is 1/8 of the input size
    '''
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    #c4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(c4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    #Transition
    '''
    **([c5, n_filters])** For giving more than 1 tensor as an input to lambda layer.
    Now this list will be passed on to the function inside lambda layer. We can call the values by normal indexing 
    operation in order. e.g.
    
    def custom_layer(tensor):
        tensor1 = tensor[0]
        tensor2 = tensor[1]
        return tensor1 + tensor2
    '''
    ctrp = Lambda(PAM, name="lambda_PAM")(c5)
    ctrc = Lambda(CAM, name="lambda_CAM")(c5)
    ctr = add([ctrp, ctrc])
    
    outputs = Conv2D(output_ch, (1, 1))(ctr)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def BAM_net(input_img, n_filters, dropout, batchnorm = True):
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    
    
    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    c4_es = BAM(c4, n_filters * 8, 3, dil_rate = 4)
    u6 = concatenate([u6, c4_es])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    c3_es = BAM(c3, n_filters * 4, 3, dil_rate = 4)
    u7 = concatenate([u7, c3_es])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    c2_es = BAM(c2, n_filters * 2, 3, dil_rate = 4)
    u8 = concatenate([u8, c2_es])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    c1_es = BAM(c1, n_filters * 1, 3, dil_rate = 4)
    u9 = concatenate([u9, c1_es])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    outputs = Conv2D(output_ch, (1, 1))(c9)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def CBAM_net(input_img, n_filters, dropout, batchnorm = True):
    c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
    c5 = Dropout(dropout)(c5)
    c5 = SE_ResNet(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    
    
    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    c4_es = CBAM(c4, n_filters * 8)
    u6 = concatenate([u6, c4_es])
    u6 = Dropout(dropout)(u6)
    c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    c3_es = CBAM(c3, n_filters * 4)
    u7 = concatenate([u7, c3_es])
    u7 = Dropout(dropout)(u7)
    c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    c2_es = CBAM(c2, n_filters * 2)
    u8 = concatenate([u8, c2_es])
    u8 = Dropout(dropout)(u8)
    c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    c1_es = CBAM(c1, n_filters * 1)
    u9 = concatenate([u9, c1_es])
    u9 = Dropout(dropout)(u9)
    c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
    outputs = Conv2D(output_ch, (1, 1))(c9)#, activation='softmax'
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def RED_net(input_img, n_filters, dropout, L = 1, batchnorm = True):
    
    loop = L
    red_ip = Lambda(Red_net_ip, name="gray_concat")(input_img)
    for i in range(loop):
        c0 = Conv2D(starting_ch, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(red_ip)
        c0 = BatchNormalization()(c0)
        c0 = Activation('relu')(c0)
        # Contracting Path
        c1 = SE_ResNet(c0, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        c1 = SE_ResNet0(c1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)  
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)
        
        c2 = SE_ResNet(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        c2 = SE_ResNet0(c2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        
        c3 = SE_ResNet(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
        c3 = SE_ResNet0(c3, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 2)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = SE_ResNet(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
        c4 = SE_ResNet0(c4, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 4)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
        c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 8)
        c5 = Dropout(dropout)(c5)
        c5 = SE_ResNet(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        c5 = SE_ResNet0(c5, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
    
        # Expanding Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = SE_ResNet(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = SE_ResNet(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = SE_ResNet(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = SE_ResNet(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm, dil_rate = 1)
        
        outputs = Conv2D(output_ch, (1, 1))(c9)#, activation='sigmoid'
        red_ip = concatenate([input_img, outputs])
        
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

#%%
def Seg_net_original(input_img, n_filters, dropout, kernel=3, batchnorm = True):
    
    # encoder
    conv_1 = Conv2D(n_filters, (kernel, kernel), padding="same")(input_img)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(n_filters, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D((2,2))(conv_2)

    conv_3 = Conv2D(n_filters*2, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(n_filters*2, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D((2,2))(conv_4)

    conv_5 = Conv2D(n_filters*4, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(n_filters*4, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(n_filters*4, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D((2,2))(conv_7)

    conv_8 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D((2,2))(conv_10)

    conv_11 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D((2,2))(conv_13)
    # decoder

    unpool_1 = MaxUnpooling2D((2,2))([pool_5, mask_5])

    conv_14 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D((2,2))([conv_16, mask_4])

    conv_17 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(n_filters*8, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(n_filters*4, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D((2,2))([conv_19, mask_3])

    conv_20 = Conv2D(n_filters*4, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(n_filters*4, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(n_filters*2, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D((2,2))([conv_22, mask_2])

    conv_23 = Conv2D(n_filters*2, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(n_filters, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D((2,2))([conv_24, mask_1])

    conv_25 = Conv2D(n_filters, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(output_ch, (1, 1), padding="valid")(conv_25)
    outputs = BatchNormalization()(conv_26)
    #outputs = Activation('softmax')(conv_26)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model






































