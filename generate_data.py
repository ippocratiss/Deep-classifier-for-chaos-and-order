from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from functions import *

################################
# This module mainly defines the functions which load the datasets (training, validation, testing),
# and the functions which produce the iterators with the neccesary data augmentation.
################################

# Defines the function that loads the training data
def load_data_SM(Np = 1000, norm = True, plot = False):
    # Np is the number of total data points in the training/validation set 
    import random
    Np1 = int(Np/11)
    Np2 = int(Np/7)        
    
    ################################
    K_order_semip0 = [np.linspace(0.0001, 0.099,  Np1), np.pi,  1.5 ] 
    K_order_semip1 = [np.linspace(0.1, 0.5,  Np1), np.pi,  1.5 ] 
    K_order_semip2 = [np.linspace(1.0, 1.04,  Np1), np.pi, 1.5 ]
    K_order_semip3 = [np.linspace(0.1, 3.5,  Np1), np.pi,   0.5 ]
    #
    K_order_islands1 = [np.linspace(1.0, 1.03,  Np1), np.pi, 2.5  ]
    K_order_islands2 = [np.linspace(1.9, 2.01,  Np1), np.pi, 1.5  ]
    K_order_islands3 = [np.linspace(1.20, 1.22,  Np1),np.pi, 1.5  ]
    #
    K_order_islands4 = [np.linspace(0.5, 0.55,  Np1),    np.pi, 4.75 ]
    K_order_islands5 = [np.linspace(0.51, 0.5108,  Np1), np.pi, 4.8  ]
    K_order_islands6 = [np.linspace(0.51, 0.59,  Np1),   np.pi, 4.85 ]
    K_order_islands7 = [np.linspace(0.59, 0.590001,  Np1),np.pi, 5.  ]
    K_order_islands8 = [0.8, np.pi, np.linspace(8, 8.6,  Np1)    ]
    K_order_islands9 = [np.linspace(0.5, 0.59,  Np1),   np.pi, 10.   ]
    
    K_order_distislands = [np.linspace(1.44, 1.465,  Np1), np.pi, 1.5]
    ################################
    K_chaos_islands1 = [np.linspace(2.0240, 2.02405,  Np2), np.pi, 1.5]
    K_chaos_islands2 = [np.linspace(1.451, 1.4565,  Np2), np.pi, 1.5]
    K_chaos_islands3 = [np.linspace(4.4, 4.6,  Np2), np.pi , 0.1]
    #
    K_chaos_islands4 = [np.linspace(0.9, 1.2,  Np2), np.pi, 8.]
    K_chaos_islands5 = [np.linspace(1.4, 1.6,  Np2), np.pi, -4.1]
    K_chaos_islands6 = [np.linspace(0.9, 1.3,  Np2), np.pi, -4.1]
    #
    K_chaos_KAM = [np.linspace(2.042, 2.049,  Np2), np.pi, 1.5]
    #
    K_chaos_global = [np.linspace(6.0, 7.0,  Np2), np.pi, 1.5]
    #
    ################################
    data_train_order = [] 
    data_train_chaos = []
    
    ###### ORDER ##########
    #######################     
    
    for i in range(0, Np1): # order: semi-periodic 
        map_i0 = Standard_map(K_order_semip0[0][i], x0 = K_order_semip0[1], y0 = K_order_semip0[2], plot = plot, norm = norm)
        map_i1 = Standard_map(K_order_semip1[0][i], x0 = K_order_semip1[1], y0 = K_order_semip1[2], plot = plot, norm = norm)
        map_i2 = Standard_map(K_order_semip2[0][i], x0 = K_order_semip2[1], y0 = K_order_semip2[2], plot = plot, norm = norm)
        map_i3 = Standard_map(K_order_semip3[0][i], x0 = K_order_semip3[1], y0 = K_order_semip3[2], plot = plot, norm = norm)
        
        data_train_order.append(map_i1)
        data_train_order.append(map_i2)
        #data_train_order.append(map_i3)

    for i in range(0, Np1): # order: islands
        map_i1 = Standard_map(K_order_islands1[0][i], x0 = K_order_islands1[1], y0 = K_order_islands1[2], plot = plot, norm = norm)
        map_i2 = Standard_map(K_order_islands2[0][i], x0 = K_order_islands2[1], y0 = K_order_islands2[2], plot = plot, norm = norm)
        map_i3 = Standard_map(K_order_islands3[0][i], x0 = K_order_islands3[1], y0 = K_order_islands3[2], plot = plot, norm = norm)
        map_i4 = Standard_map(K_order_islands4[0][i], x0 = K_order_islands4[1], y0 = K_order_islands4[2], plot = plot, norm = norm)
        map_i5 = Standard_map(K_order_islands5[0][i], x0 = K_order_islands5[1], y0 = K_order_islands5[2], plot = plot, norm = norm)
        map_i6 = Standard_map(K_order_islands6[0][i], x0 = K_order_islands6[1], y0 = K_order_islands6[2], plot = plot, norm = norm)
        map_i7 = Standard_map(K_order_islands7[0][i], x0 = K_order_islands7[1], y0 = K_order_islands7[2], plot = plot, norm = norm)
        map_i8 = Standard_map(K_order_islands8[0], x0 = K_order_islands8[1],    y0 = K_order_islands8[2][i], plot = plot, norm = norm)
        map_i9 = Standard_map(K_order_islands9[0][i], x0 = K_order_islands9[1], y0 = K_order_islands9[2], plot = plot, norm = norm)
        data_train_order.append(map_i1)
        data_train_order.append(map_i2)
        data_train_order.append(map_i3)
        data_train_order.append(map_i4)
        data_train_order.append(map_i5)
        data_train_order.append(map_i6)
        data_train_order.append(map_i7)
        data_train_order.append(map_i8)
        data_train_order.append(map_i9)
        
    ###### CHAOS ##########
    #######################
    
    for i in range(0,  Np2): # chaos: islands + chaos
        map_i1 = Standard_map(K_chaos_islands1[0][i], x0 = K_chaos_islands1[1], y0 = K_chaos_islands1[2], plot = plot, norm = norm)
        map_i2 = Standard_map(K_chaos_islands2[0][i], x0 = K_chaos_islands2[1], y0 = K_chaos_islands2[2], plot = plot, norm = norm)
        map_i3 = Standard_map(K_chaos_islands3[0][i], x0 = K_chaos_islands3[1], y0 = K_chaos_islands3[2], plot = plot, norm = norm)
        map_i4 = Standard_map(K_chaos_islands4[0][i], x0 = K_chaos_islands4[1], y0 = K_chaos_islands4[2], plot = plot, norm = norm)
        map_i5 = Standard_map(K_chaos_islands5[0][i], x0 = K_chaos_islands5[1], y0 = K_chaos_islands5[2], plot = plot, norm = norm)
        data_train_chaos.append(map_i1)
        data_train_chaos.append(map_i2)
        data_train_chaos.append(map_i3)
        data_train_chaos.append(map_i4)
        data_train_chaos.append(map_i5)
        
    for i in range(0,  Np2): # chaos: KAM structures surrounded by chaos
        map_i = Standard_map(K_chaos_KAM[0][i], x0 = K_chaos_KAM[1], y0 = K_chaos_KAM[2], plot = plot, norm = norm)
        data_train_chaos.append(map_i)
    
    for i in range(0,  Np2): # chaos: global
        map_i = Standard_map(K_chaos_global[0][i], x0 = K_chaos_global[1], y0 = K_chaos_global[2], plot = plot, norm = norm)
        data_train_chaos.append(map_i)                        
    #######################
    #######################
    
    data_train_order = np.array(data_train_order)  
    data_train_chaos = np.array(data_train_chaos) 
    order_labels = np.zeros(len(data_train_order))
    chaos_labels = np.ones(len(data_train_chaos))

    data_train = np.concatenate((data_train_order, data_train_chaos ), axis=0) #np.concatenate((data_test_order, data_test_chaos ), axis=0)
    data_label = np.concatenate((order_labels, chaos_labels ), axis=0)
    if len(data_train) != len(data_label):
        print('Length of train data != label data')
    return data_train, data_label


# Loads the benchmark testing data set
def load_data_deVog(Np = 400, batch_size=64, norm = True, noise = 0, plot = False):
    import random
    Np1 = int(Np/7)
    Np2 = int(Np/4)     
    
    # deformed semiperiodic orbit: (-0.08 - -0.1, 0.2, 0.), (-0.8 - -0.99, 0.2, 0.)
    # islands order: 
    #         (0.098 --> 0.11, 0.45, 0.), (0.0005 --> 0.0007, 0.45, 0.),(0.7399 --> 0.74, 0.25, 0.)
    #         (-0.005 --> -0.02, 0.2, 0.), (-1.04 --> -1.06, 0.2, 0.) 
    #         [0.5, 0.2, 0.2] --> [0.51, 0.2, 0.2]
    #         [-1.0515, -0.1, -0.005] --> [-1.053, -0.1, -0.005]
    #         [-1.06, -0.1, 0.005] --> [-1.0601, -0.1, 0.005]
    # islands chaos: 
    #         (-1.03 --> -1.06, 0.01, 0.), (-1.0598 --> -1.0599, 0.1, 0.)
    #         [-1.02, 0.0, 0.004] --> [-1.05, 0.0, 0.004]
    #         [-1.02, 0.0, 0.00004] --> [-1.05, 0.0, 0.00004]
    ################################

    K_order_semip1 = [np.linspace(0.01, 0.2,  Np1), 0.01,  0. ] 
    #
    K_order_islands1 = [np.linspace(0.10, 0.11,  Np1),    0.45, 0.   ]
    K_order_islands2 = [np.linspace(0.7399, 0.74,  Np1),   0.25, 0.   ]
    K_order_islands3 = [np.linspace(-0.005, -0.02,  Np1),   0.2, 0.  ]
    K_order_islands4 = [np.linspace(-1.04, -1.06,  Np1),   0.2, 0.  ]
    K_order_islands5 = [np.linspace(-1.0515, -1.053,  Np1), -0.1, -0.005 ]
    K_order_islands6 = [np.linspace(-1.06, -1.0601,  Np1),  -0.1, 0.005]
    ################################
    K_chaos_islands1 = [np.linspace(-1.03, -1.05,  Np2), 0.01, 0. ]
    K_chaos_islands2 = [np.linspace(-1.0598, -1.0599,  Np2), 0.1,0. ]
    K_chaos_islands3 = [np.linspace(-1.02, -1.05,  Np2), 0. , 0.004]
    K_chaos_islands4 = [np.linspace(-1.02, -1.05,  Np2), 0. , 0.00004]
    ################################
    
    data_order = [] 
    data_chaos = []

    for i in range(0, Np1): # order: semi-periodic 
        map_i1 = deVog_map(K_order_semip1[0][i], x0 = K_order_semip1[1], y0 = K_order_semip1[2], plot = plot, norm = norm)      
        data_order.append(map_i1)

    for i in range(0, Np1): # order: islands
        map_i1 = deVog_map(K_order_islands1[0][i], x0 = K_order_islands1[1], y0 = K_order_islands1[2], plot = plot, norm = norm)
        map_i2 = deVog_map(K_order_islands2[0][i], x0 = K_order_islands2[1], y0 = K_order_islands2[2], plot = plot, norm = norm)
        map_i3 = deVog_map(K_order_islands3[0][i], x0 = K_order_islands3[1], y0 = K_order_islands3[2], plot = plot, norm = norm)
        map_i4 = deVog_map(K_order_islands4[0][i], x0 = K_order_islands4[1], y0 = K_order_islands4[2], plot = plot, norm = norm)
        map_i5 = deVog_map(K_order_islands5[0][i], x0 = K_order_islands5[1], y0 = K_order_islands5[2], plot = plot, norm = norm)
        map_i6 = deVog_map(K_order_islands6[0][i], x0 = K_order_islands6[1], y0 = K_order_islands6[2], plot = plot, norm = norm)
        data_order.append(map_i1)
        data_order.append(map_i2)
        data_order.append(map_i3)
        data_order.append(map_i4)
        data_order.append(map_i5)
        data_order.append(map_i6)
   
    ################
    for i in range(0,  Np2): # chaos: islands + chaos
        map_i1 = deVog_map(K_chaos_islands1[0][i], x0 = K_chaos_islands1[1], y0 = K_chaos_islands1[2], plot = plot, norm = norm)
        map_i2 = deVog_map(K_chaos_islands2[0][i], x0 = K_chaos_islands2[1], y0 = K_chaos_islands2[2], plot = plot, norm = norm)
        map_i3 = deVog_map(K_chaos_islands3[0][i], x0 = K_chaos_islands3[1], y0 = K_chaos_islands3[2], plot = plot, norm = norm)
        map_i4 = deVog_map(K_chaos_islands4[0][i], x0 = K_chaos_islands4[1], y0 = K_chaos_islands4[2], plot = plot, norm = norm)
        data_chaos.append(map_i1)
        data_chaos.append(map_i2)
        data_chaos.append(map_i3)
        data_chaos.append(map_i4)     
    ################
    
    data_order = np.array(data_order)  
    data_chaos = np.array(data_chaos) 
    order_labels = np.zeros(len(data_order))
    chaos_labels = np.ones(len(data_chaos))
    data_deVog = np.concatenate((data_order, data_chaos ), axis=0) 
    data_label = np.concatenate((order_labels, chaos_labels ), axis=0)    
    if len(data_deVog) != len(data_label):
        print('Length of train data != label data')
 
    return data_deVog, data_label 

# Produces the augmented training data given a training dataset. 
# The input variable "data" must be in the form data = data values, labels.
def produce_iterator(data, batch_size=64, print_stats=False,
                       shuffle=False, rotation_range=0, width_shift_range=0,    
                       height_shift_range=0, shear_range=0, fill_mode='nearest'):
# load data with the approapriate data-loading function, 'load_XXXX'.
# frac: fraction of training data to use for training. Validation data are then (1-frac) of that.
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range, 
            height_shift_range=height_shift_range, 
            shear_range=shear_range,                             
            fill_mode=fill_mode                                   
            )    

    (dataX, datay) = data
    width, height, channels = dataX.shape[1], dataX.shape[2], 1
    dataX = dataX.reshape((dataX.shape[0], width, height, channels))

    ## fit the image generators to the data
    datagen.fit(dataX)
    ## define iterators
    iterator = datagen.flow(dataX, datay, batch_size=batch_size, shuffle=shuffle)

    ## define batches
    batchX, batchy = iterator.next()
    if print_stats == True:
        print('Dataset statistics: ',   batchX.shape, batchX.mean(), batchX.std())
    return iterator


# Plots the statistics of the given data set
def train_data_stats(iterator_set, label):
    #(trainX, trainy) = data_set
    iterator = iterator_set   
    #iterator = datagen.flow(trainX, trainy, batch_size=64, shuffle=True)
    batchX, batchy = iterator.next()
    print(label + ' ' + 'statistics (shape, mean, std) :', batchX.shape, batchX.mean(), batchX.std())
    return batchX.mean(), batchX.std() 


def load_data_Web(Np = 200, batch_size=16, norm = True, noise = 0, plot = False, 
                  chaos_data_only = False, order_data_only = False):
    import random
    Np1 = int(Np/1)
    Np2 = int(Np/1)             
    data_order = [] 
    data_chaos = []
    K_order = np.linspace(0.01, 1.,  Np2) # We vary only K here. Initial (x,y) are the default ones.
    K_chaos = np.linspace(2.1, 2.9,  Np2) # We vary only K here. Initial (x,y) are the default ones.
    
    for i in range(0, Np1): # order: semi-periodic 
        map_i1 = Web_map(K_order[i],  plot = plot, norm = norm)      
        data_order.append(map_i1)         
    ################
    for i in range(0,  Np2): # chaos: web structures
        map_i1 = Web_map(K_chaos[i], plot = plot, norm = norm)
        data_chaos.append(map_i1)     
    
    data_order = np.array(data_order)  
    data_chaos = np.array(data_chaos) 
    order_labels = np.zeros(len(data_order))
    chaos_labels = np.ones(len(data_chaos))
    data_Web = np.concatenate((data_order, data_chaos ), axis=0) 
    data_label = np.concatenate((order_labels, chaos_labels ), axis=0)
    if chaos_data_only == True:
        data_Web, data_label = data_chaos, chaos_labels
    if order_data_only == True:
        data_Web, data_label = data_order, order_labels
    if len(data_Web) != len(data_label):
        print('Length of train data != label data')
 
    return data_Web, data_label 
