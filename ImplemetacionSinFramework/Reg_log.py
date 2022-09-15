#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_error_ = [] 
Params_final = []

#hypothesis function
def hypothesis(param, sample):
    #param: list with all the parameters
    #sample: a single row of the dataset 
    #sigma: the prediction for that row
    aux = 0
    num_param = len(param)
    for i in range (num_param):  #for each parameter
        aux = (param[i]*sample[i]) + aux  #mx+b (m1x1+m2x2+b)
    aux = aux*(-1)
    sigma = 1/(1+(np.exp(aux))) #sigmoid 
    return sigma  #probability that the sample is part of a category 

#Cost function
def error(real, samples, param):
    #real: the real values for y
    #sample: the data set with samples
    #param: list with the parameters
    error = 0
    error1 = []
    acum = 0
    hyp = []
    num_samples = len(samples)
    for i in range(num_samples):
        h = hypothesis(param,samples[i]) #prediction
        hyp.append(h)
        
        #identify one class:
        if real[i] == 1:
            if h == 0:
                h = 0.0001
            error = np.log(h);
            error = (-1)*error
            error1.append(error)
        if real[i] == 0:
            if h == 1:
                h = 0.9999
            error = np.log(1-h)
            error = (-1)*error
            error1.append(error)
        
        #print error the prediction and the real value for y
        acum = acum + error
        print( "error %f  h  %f  real %f " % (error, h,  real[i])) 
    
    mean_error = acum/num_samples
    _error_.append(mean_error)
    return mean_error

#Optimization function
def gradiant(params,samples,real,a):
    p = list(params)
    num_params = len(params)
    num_samples = len(samples)
    aux = (-1)*(a/num_samples) #-learning rate/number of samples
    for i in range(num_params): #Sum in the formula
        sum = 0
        for j in range (num_samples):
            error = hypothesis(params,samples[j])-real[j]
            sum = sum + error*samples[j][i] 
        p[i] = params[i] + aux*sum
        
    return p

#Logistic Regression 
def re_log(alfa,samples,y):
    params = [0,0,0]
    while True:
        oldparams = list(params) #important to save the old parameters
        params = gradiant(params,samples,y,alfa)	
        errors = error(y, samples, params)  
        if(oldparams == params or errors < 0.7): # local minima is found when there is no further improvement or stop when error is 0 
            break
    plt.plot(_error_)
    plt.title("Error of one class")

    return params


#load data
columns = ["class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids",
"Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df = pd.read_csv('wine.data',names=columns)

#clean data
df_clean = df.drop(["Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids",
"Nonflavanoid phenols","Proanthocyanins","Hue","OD280/OD315 of diluted wines","Proline"], axis=1)
samples = df_clean[["Alcohol","Color intensity"]].to_numpy()
samples = np.c_[samples, np.zeros(178)]

#one hot encoding
aux = df_clean["class"].to_numpy()
y = np.zeros((178,3), dtype=np.int8)
for i in range(178):
    if aux[i] == 1:
        y[i][0] = 1
    elif aux[i] == 2:
        y[i][1] = 1
    elif aux[i] == 3:
        y[i][2] = 1

alfa = 0.03 #learning rate

#the data set has 3 clases, we get the parameters for each class
for i in range(3):
    new_params = re_log(alfa,samples,y[:,i])
    print("Parameters for class ",i+1)
    print(new_params)
    Params_final.append(new_params)
    plt.show()