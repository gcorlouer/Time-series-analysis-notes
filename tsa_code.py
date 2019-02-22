import numpy as np
import random 
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt 
#1D Random walk
N=500 #number of observations
#Initialise
X=np.zeros((1,N), float)
epsilon=np.random.normal(loc=0.0, scale=1, size=(1,N))
X[0,0]=epsilon[0,0]
#Generate Random Walk
for i in range(N-1):
    X[0,i+1]=X[0,i]+epsilon[0,i]
plt.plot(X[0,:])
#ARMA(2,2)
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]#step is zero is unweighted
maparams = np.r_[1, maparams]#step is zero is unweighted
y = arma_generate_sample(arparams, maparams, 500)#mean zero variance 1
plt.plot(y)
#AR(1)
#Stable
X=np.zeros((1,N), float)
epsilon=np.random.normal(loc=0.0, scale=1, size=(1,N))
X[0,0]=epsilon[0,0]
for i in range(N-1):
    X[0,i+1]=0.5*X[0,i]+epsilon[0,i]
plt.plot(X[0,:])
#Unstable
for i in range(N-1):
    X[0,i+1]=2*X[0,i]+epsilon[0,i]
plt.plot(X[0,:])
#Estimate ACF and PACF
acf_X=acf(np.squeeze(X), nlags=10)
pacf_X=pacf(np.squeeze(X), nlags=10)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(np.squeeze(X), lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(np.squeeze(X), lags=10, ax=ax2)
