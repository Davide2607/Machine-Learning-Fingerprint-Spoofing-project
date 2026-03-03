import numpy as np
import matplotlib.pyplot as plt

def mrow(col):
    return col.reshape((1,col.size))

def mcol(row):
    return row.reshape((row.size,1))

def createCov(D):   
    mu = 0
    C = 0
    mu = D.mean(1)
    for i in range(D.shape[1]):
        C = C + np.dot(D[:, i:i+1] - mu, (D[:, i:i+1] - mu).T)  #scalar product using numpy
        #with this formule we have just centered the data (PCA on NON CENTERED DATA is quite an unsafe operation) 
    
    C = C / float(D.shape[1])   #where the divider is the dimension N of our data 
    return C

def createCenteredCov(DC):      #for centered data yet
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    
    return C

def centerData(D):
    mu = D.mean(1)
    DC = D - mcol(mu)   
    return DC

#logpdf_GAU_ND algorithm for an array(just one sample at time)
def logpdf_GAU_1Sample(x,mu,C):  
   
    xc = x-mu
    M = x.shape[0]
    logN = 0
    const = - 0.5 * M *np.log(2*np.pi)
    log_determ = np.linalg.slogdet(C)[1]
    lamb = np.linalg.inv(C)
    last_elem = np.dot(xc.T,np.dot(lamb,xc)).ravel()
    logN = const - 0.5 * (log_determ) - 0.5* (last_elem)

    return logN

def logpdf_GAU_ND(X,mu,C):          #logpdf_GAU_ND algorithm for a Multi-D matrix
    logN = []

    for i in range(X.shape[1]):

        single_sample = logpdf_GAU_1Sample(X[:,i:i+1],mu,C)
        logN.append(single_sample) 
    return np.array(logN)

  
def conf_plot(feauture0,feauture1,m0, C0,m1,C1):
    
    #plt.figure()
    plt.hist(feauture0.ravel(),bins = 50 ,density=True,label = 'Fake fingerprint')
    plt.hist(feauture1.ravel(),bins = 50 ,density=True, label = 'Authentic fingerprint')
    
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(mrow(XPlot), m0, C0)), color='Green', label ='Fake')
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(mrow(XPlot), m1, C1)), color='Blue',label= 'Authentic')
    plt.legend()
    plt.show()




        
        
        