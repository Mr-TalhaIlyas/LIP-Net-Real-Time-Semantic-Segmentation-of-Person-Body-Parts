import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tensorflow.python.client import device_lib
if tf.__version__ == '2.2.0' or tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0-rc2':
    import tensorflow.keras.backend as K
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    from keras import backend as K


#%%
def get_available_gpus():
    """ Get available GPU devices info. """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#%%
def model_profile(model, Batch_size, initial_lr, w_decay_value, init_dropout, 
                  lr_schedule, Weight_Decay, use_mydropout, dropout_schedule, 
                  dropout_after_epoch, LOSS_function, batchX, batchY):
    
    model_profile = np.array(['No. of FLOPs', 'No. of parameters', 'Memory for storing Parameters',
                          'Memory Required by Complete Model', 'Using GPU(s)', 'Input Batch Shape',
                          'Labels Batch Shape', 'Model Output Shape', 'Initial LR', 'LR Decay',
                          'Weight Decay', 'Weight Decay Coff.','Initial Dropout','Dropout Decay', 
                          'Dropout Schedule', 'Dropout after Epoch','Loss Finction' ]).reshape(-1,1)
    profile_values = np.empty((len(model_profile),1))
    value_unit = np.array(['GFLOPs', 'Million', 'MB', 'GB', 'GPU(s)', 'BxHxWxC', 'BxHxWxC', '?xHxWxC','-'
                           ,'-','-','-','-','-','-','-','-']).reshape(-1,1)
    try:
        _, layer_flops, _, _ = profile(model)
    except:
        layer_flops = 0
    trainable_param = model.count_params()
    memory_param = (trainable_param*16)/(8*1024**2)
    try:
        memory = get_model_memory_usage(Batch_size, model)
    except:
        memory = 0
        
    profile_values[0] = np.sum(layer_flops)/10**9
    profile_values[1] = trainable_param/10**6
    profile_values[2] = memory_param
    profile_values[3] = memory
    try:
        profile_values[4] = len(tf.config.experimental.list_physical_devices('GPU'))
    except:
        profile_values[4] = len(get_available_gpus())
    profile_values[5] = 0
    profile_values[6] = 0
    profile_values[7] = 0
    profile_values[8] = initial_lr
    profile_values[9] = 0
    profile_values[10] = 0
    profile_values[11] = w_decay_value
    profile_values[12] = init_dropout
    profile_values[13] = 0
    profile_values[14] = 0
    profile_values[15] = 0
    profile_values[16] = 0
    model_profile = np.concatenate((model_profile, profile_values, value_unit), 1)
    model_profile[5,1] = batchX
    model_profile[6,1] = batchY
    model_profile[7,1] = str(model.output.shape)
    model_profile[9,1] = lr_schedule
    model_profile[10,1] = Weight_Decay
    model_profile[13,1] = bool(use_mydropout)
    model_profile[14,1] = str(dropout_schedule)
    model_profile[15,1] = str(dropout_after_epoch)
    model_profile[16,1] = str(LOSS_function)
    table = tabulate(np.ndarray.tolist(model_profile), headers = ["Metric", "Value", 'Unit'], tablefmt="github")
    
    return table
    





#%%
def param_count(model):
    
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count, non_trainable_count

def memory_usage(model):
    
    shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))    
    memory = shapes_count * 4
    return memory
#%%
    
def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
#%%

# bunch of cal per layer
def count_linear(layers):
    MAC = layers.output_shape[1] * layers.input_shape[1]
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[1]
    else:
        ADD = 0
    return MAC*2 + ADD

def count_conv2d(layers, log = False):
    if log:
        print(layers.get_config())
    # number of conv operations = input_h * input_w / stride = output^2
    numshifts = int(layers.output_shape[1] * layers.output_shape[2])
    
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[3]
    else:
        ADD = 0
        
    return MACperConv * numshifts * 2 + ADD

def profile(model, log = False):
    # make lists
    layer_name = []
    layer_flops = []
    # TODO: relus
    inshape = []
    weights = []
    # run through models
    for layer in model.layers:
        if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"]:
            layer_flops.append(count_linear(layer))
            layer_name.append(layer.get_config()["name"])
            inshape.append(layer.input_shape)
            weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
        elif "conv" in layer.get_config()["name"] and "pad" not in layer.get_config()["name"] and "bn" not in layer.get_config()["name"] and "relu" not in layer.get_config()["name"] and "concat" not in layer.get_config()["name"]:
            layer_flops.append(count_conv2d(layer,log))
            layer_name.append(layer.get_config()["name"])
            inshape.append(layer.input_shape)
            weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
        elif "res" in layer.get_config()["name"] and "branch" in layer.get_config()["name"]:
            layer_flops.append(count_conv2d(layer,log))
            layer_name.append(layer.get_config()["name"])
            inshape.append(layer.input_shape)
            weights.append(int(np.sum([K.count_params(p) for p in set(layer.trainable_weights)])))
            
    return layer_name, layer_flops, inshape, weights