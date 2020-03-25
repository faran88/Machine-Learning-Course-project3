# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:00:38 2020

@author: faranak abri
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt 
"""------------------------------------------------------"""
np.random.seed(44)
L=100 # #of datasets
N=25
n_test=1000
lnlam= np.arange(-3,3,0.1)
lam=np.exp(lnlam)
Ms=np.random.uniform(0,1,N)
degree=25

X=np.random.uniform(0,1,N)
t=np.zeros((N,L))
h=np.sin(2*np.pi*X)

x_test=np.random.uniform(0,1,n_test)
t_test=np.sin(2*np.pi*x_test)+np.random.normal(0,0.3,n_test)
h_test=np.sin(2*np.pi*x_test)   
    
fits=np.zeros((N,L,len(lam)))
w=np.zeros((degree+1,L,len(lam)))

fits_test=np.zeros((n_test,L,len(lam)))

for i in range (L):
    t[:,i]=np.sin(2*np.pi*X)+np.random.normal(0,0.3,N)

phi=np.ones((N,degree+1))
phi_test=np.ones((n_test,degree+1))
for d in range (degree):
    phi[:,d+1]=np.exp(-(pow((X-Ms[d-1]),2)/0.02))
    phi_test[:,d+1]=np.exp(-(pow((x_test-Ms[d-1]),2)/0.02))

I=np.identity(degree+1)

for k in range (len(lam)):
    for l in range (L):       
        w[:,l,k]=(np.linalg.pinv((phi.T@phi)+(lam[k]*I)))@(phi.T@t[:,l])
        fits[:,l,k]=phi@w[:,l,k]
        fits_test[:,l,k]=phi_test@w[:,l,k]
        
fhat_test=(1/L)*(np.sum(fits_test,axis=1))
error_test=(1/n_test)*np.sum((pow((fhat_test-t_test.reshape(n_test,1)),2)),axis=0)

fhat=(1/L)*(np.sum(fits,axis=1))
bias2=(1/N)*np.sum(pow((fhat-h.reshape(N,1)),2),axis=0)
variance=(1/N)*np.sum((1/L)*np.sum(pow((fits-fhat[:,np.newaxis,:]),2),axis=1),axis=0)
bi2var=bias2+variance

plt.figure()
plt.xlabel(r"$\ln $ $\lambda$")
plt.plot(lnlam,bias2,color='blue', label=r"$(bias)^2$")
plt.plot(lnlam,variance,color='red', label="variance")
plt.plot(lnlam,bi2var,color='violet', label=r"$(bias)^2+variance$")
plt.plot(lnlam,error_test,color='black', label="test error")
plt.legend(loc='upper') 
plt.show() 
