import numpy as np
from scipy import rand

################################
# This module defines the generating functions for the Poincare maps, from which the 
# order/chaos data sets will be generated. In the maps, K/α/c are the respective 
# perturbation parameters away from integrability (real number). x0, y0 are the initial conditions. 
################################

n_points = 450 # number of evaluations points for x and y. Total points of map = 2*n_points.
res = int(np.sqrt(2*n_points)) # must be integer. This will be the resolution of the 'image'.

# Standard map  -- computes the Poincare map and returns a set of points (x,y).
def Standard_map(K, n_points = n_points, x0 = np.pi, y0 = 0.1, plot = False, norm = False, noise = 0):
    x = np.zeros(n_points) 
    y = np.zeros(n_points)   
    data_x = np.array([]) 
    data_y = np.array([]) 
    data_Standard_map = []
    x[0] = x0 # initial point for x 
    y[0] = y0 # initial point for y
    for n in range (1, n_points): # iterate to compute the map
        y[n]= (y[n-1] + K*np.sin(x[n-1])) 
        x[n]= (x[n-1] + y[n])
        y[n] = y[n]%(2*np.pi)
        x[n] = x[n]%(2*np.pi)
        if y[n] > np.pi: # add modulo 
            y[n] = y[n] - 2*np.pi
        err1 = noise*np.random.normal(1) # noise for x[n]
        err2 = noise*np.random.normal(1) # noise for y[n]
        x[n] = x[n] + err1 
        y[n] = y[n] + err2
    data_x = x
    data_y = y
        #data_x.append(x[n])
        #data_y.append(y[n])
        #data_x = np.array(data_x)
        #data_y = np.array(data_y)
    if norm == True: # normalise x,y between [0,1]
        final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if norm ==False:
        final_data = np.array([ data_x, data_y ])
    if plot == True: # tabulate data as pairs (x,y) - convenient for plotting the orbit
        return final_data
    if plot == False: # tabulate results as image (res)x(res) - input for the ML network
        final_data = np.array([data_x/max(np.abs(data_x)) , data_y/max(np.abs(data_y))  ])
        final_data_2 = final_data.reshape(res,res)
        return final_data_2
    
# de Vogelare Poincare map - c is the perturbation parameter
def deVog_map(c, n_points = n_points, noise = 0, x0 = 0.01, y0 = 0, plot = False, norm = False  ):
    x=np.zeros(n_points)
    y=np.zeros(n_points)
    ε = 0.
    data_x = [] 
    data_y = [] 
    data_deVog = []
    x[0] = x0
    y[0] = y0
    for n in range(1,n_points):
        err1 = noise*np.random.normal(1)
        err2 = noise*np.random.normal(1)
        x[n] = -y[n-1] + c*x[n-1] + x[n-1]**2
        y[n ]=  x[n-1] - (c*x[n] + x[n]**2)
        x[n] = x[n] + err1
        y[n] = y[n] + err2
    data_x = x
    data_y = y
        #data_x.append(x[n])
        #data_y.append(y[n])
        #data_x = np.array(data_x)
        #data_y = np.array(data_y)
    final_data = np.array([ data_x/(ε+max(np.abs(data_x))), data_y/(ε+max(np.abs(data_y))) ])
    if norm == True:
        final_data = np.array([ data_x/(ε+max(np.abs(data_x))), data_y/(ε+max(np.abs(data_y))) ])
    if norm ==False:
        final_data = np.array([ data_x, data_y ])
    if plot == True:
        return final_data
    if plot == False: 
        final_data = np.array([ data_x/(ε+max(np.abs(data_x))), data_y/(ε+max(np.abs(data_y))) ])
        final_data_2 = final_data.reshape(res,res)
        return final_data_2

# Web map -  α is the perturbation parameter, and q another real parameter for this system (see paper).
def Web_map(α, q = 4, n_points = n_points, x0 = 0.001, y0 = 0.001, plot = False, norm = False ):
    x=np.zeros(n_points)
    y=np.zeros(n_points)
    data_x = [] 
    data_y = [] 
    data_Web = []
    x[0] = x0
    y[0] = y0
    for n in range(1,n_points):
        x[n] =  (x[n-1] + α*np.sin(y[n-1]))*np.cos(2*np.pi/q) + y[n-1]*np.sin(2*np.pi/q) 
        y[n] = -(x[n-1] + α*np.sin(y[n-1]))*np.sin(2*np.pi/q) + y[n-1]*np.cos(2*np.pi/q) 
    data_x = x
    data_y = y
    if norm == True:
        final_data = np.array([ data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y)) ])
    if norm ==False:
        final_data = np.array([ data_x, data_y ])
    if plot == True:
        return final_data
    if plot == False:
        final_data = np.array([data_x/max(np.abs(data_x)), data_y/max(np.abs(data_y))])
        final_data_2 = final_data.reshape(res,res)
        return final_data_2