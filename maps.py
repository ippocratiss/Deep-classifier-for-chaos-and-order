import numpy as np
from matplotlib import pyplot
from scipy import rand
import tensorflow as tf
import keras 
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense

n_points = 450 #450 # number of evaluations points for x and y. Total points of map = 2*n_points.
res = int(np.sqrt(2*n_points)) # must be integer. This will be the resolution of the 'image'


def Standard_map(K, n_points = n_points, x0 = np.pi, y0 = 0.1, plot = False, noise = 10**-3):
    x = np.zeros(n_points + 1)
    y = np.zeros(n_points + 1)   
    data_x = [] 
    data_y = [] 
    data_Standard_map = []
    x[0] = x0
    y[0] = y0
    for n in range (1, n_points + 1):
        y[n]= (y[n-1] + K*np.sin(x[n-1]))
        x[n]= (x[n-1] + y[n])
        y[n] = y[n]%(2*np.pi)
        x[n] = x[n]%(2*np.pi)
        if y[n] > np.pi:
            y[n] = y[n] - 2*np.pi
        err1 = noise*np.random.normal(1)
        err2 = noise*np.random.normal(1)
        x[n] = x[n] + err1
        y[n] = y[n] + err2
        data_x.append(x[n])
        data_y.append(y[n])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if plot == True:
        return final_data
    if plot == False:
        final_data = np.array([data_x/max(np.abs(data_x)) , data_y/max(np.abs(data_y))  ])
        final_data_2 = final_data.reshape(res,res)
        #final_data_2 = np.expand_dims(final_data_2 , axis=0) 
        return final_data_2
    
    
def deVog_map(c, n_points = n_points, x0 = 0.01, y0 = 0, plot = False ):
    x=np.zeros(n_points+1)
    y=np.zeros(n_points+1)
    data_x = [] 
    data_y = [] 
    data_deVog = []
    x[0] = x0
    y[0] = y0
    for n in range(1,n_points+1):
        err1 = 0*(10**-6)*np.random.normal(1)
        err2 = 0*(10**-6)*np.random.normal(1)
        x[n]= -y[n-1] + c*x[n-1] + x[n-1]**2
        y[n]=  x[n-1] - (c*x[n] + x[n]**2)
        x[n] = x[n] + err1
        y[n] = y[n] + err2
        if np.abs(x[n]) < 10**4. and np.abs(y[n]) < 10**4.:
            data_deVog.append([x[n],y[n]])
            data_x.append(x[n])
            data_y.append(y[n])
        else:
            break                        
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if plot == True:
        return final_data
    if plot == False: 
        final_data = np.array([data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y))])
        final_data_2 = final_data.reshape(res,res)
        #final_data_2 = np.expand_dims(final_data_2 , axis=0) 
        return final_data_2

    
    
def Henon_map(α, n_points = n_points, x0 = 0.01, y0 = 0., plot = False ):
    x=np.zeros(n_points+1)
    y=np.zeros(n_points+1)
    data_x = [] 
    data_y = [] 
    data_Henon = []
    x[0] = x0
    y[0] = y0
    for n in range(1,n_points+1):
        x[n]= x[n-1]*np.cos(α) - (y[n-1] - x[n-1]**2)*np.sin(α) 
        y[n]= x[n-1]*np.sin(α) + (y[n-1] - x[n-1]**2)*np.cos(α)
        if x[n]> 0 and y[n]>0 and np.abs(x[n]) < 10**4 and np.abs(y[n]) < 10**4:
            data_Henon.append([x[n],y[n]])
        data_x.append(x[n])
        data_y.append(y[n])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if plot == True:
        return final_data
    if plot == False:
        final_data = np.array([data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y))])
        final_data_2 = final_data.reshape(res,res)
        return final_data_2
    
    
def Quadratic_map(α, n_points = n_points, x0 = 0.001, y0 = 0., plot = False ):
    x=np.zeros(n_points+1)
    y=np.zeros(n_points+1)
    data_x = [] 
    data_y = [] 
    data_Quadratic = []
    x[0] = x0
    y[0] = y0
    for n in range(1,n_points+1):
        x[n] = x[n-1] #1 - y[n-1] - α*x[n-1]**2 
        y[n] = y[n-1] + 2*α*x[n-1] #x[n-1]
        y[n] = y[n]%(2*np.pi)
        data_Quadratic.append([x[n],y[n]])
        data_x.append(x[n])
        data_y.append(y[n])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if plot == True:
        return final_data
    if plot == False:
        final_data = np.array([data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y))])
        final_data_2 = final_data.reshape(res,res)
        return final_data_2
    
    
def Web_map(α, q = 4, n_points = n_points, x0 = 0., y0 = 0., plot = False ):
    x=np.zeros(n_points+1)
    y=np.zeros(n_points+1)
    data_x = [] 
    data_y = [] 
    data_Web = []
    x[0] = x0
    y[0] = y0
    for n in range(1,n_points+1):
        x[n] =  (x[n-1] + α*np.sin(y[n-1]))*np.cos(2*np.pi/q) + y[n-1]*np.sin(2*np.pi/q) 
        y[n] = -(x[n-1] + α*np.sin(y[n-1]))*np.sin(2*np.pi/q) + y[n-1]*np.cos(2*np.pi/q) 
        data_Web.append([x[n],y[n]])
        data_x.append(x[n])
        data_y.append(y[n])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if plot == True:
        return final_data
    if plot == False:
        final_data = np.array([data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y))])
        final_data_2 = final_data.reshape(res,res)
        return final_data_2