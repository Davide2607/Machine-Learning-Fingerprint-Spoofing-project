# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:46:35 2024

@author: Davide
"""

import numpy as np
import scipy.optimize
from Evaluations import * 
from prettytable import PrettyTable

def mrow(col):
    return col.reshape((1,col.size))

def mcol(row):
    return row.reshape((row.size,1))

def mean_and_covarianceMatrix(D):
    """Compute mean and covariance matrix for a given Dataset D where each column is a sample"""
    N = D.shape[1]  
    mu = mcol(D.mean(1))
    DC = D - mu 
    C = np.dot(DC, DC.T)/N  
    return mu, C

def logpdf_GAU_ND(X,mu,C) :
    
    res = -0.5*X.shape[0]*np.log(2*np.pi)
    res += -0.5*np.linalg.slogdet(C)[1]
    res += -0.5*((X-mu)*np.dot(np.linalg.inv(C), (X-mu))).sum(0) 
    return res

def logpdf_GMM(X, gmm):
    
    SJ = np.zeros((len(gmm),X.shape[1]))
    
    for g, (w, mu, C) in enumerate(gmm):
        SJ[g,:] = logpdf_GAU_ND(X, mu, C) + np.log(w)

    SM = scipy.special.logsumexp(SJ, axis=0)
    
    return SJ, SM 

def GMM_EM(X, gmm):
    '''
    EM algorithm for GMM full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    N = X.shape[1]
    
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        for g in range(len(gmm)):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = np.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = np.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew

    return gmm


def GMM_EM_diag(X, gmm):
    '''
    EM algorithm for GMM diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    N = X.shape[1]
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        for g in range(len(gmm)):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = np.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            #diag
            Sigma = Sigma * np.eye(Sigma.shape[0])
            U, s, _ = np.linalg.svd(Sigma)
            s[s<psi] = psi
            sigma = np.dot(U, mcol(s)*U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
    return gmm


def GMM_EM_tied(X, gmm):
    '''
    EM algorithm for GMM tied covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    '''
    llNew = None
    llOld = None
    N = X.shape[1]
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        sigmaTied = np.zeros((X.shape[0],X.shape[0]))
        for g in range(len(gmm)):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (mrow(gamma)*X).sum(1)
            S = np.dot(X, (mrow(gamma)*X).T)
            w = Z/N
            mu = mcol(F/Z)
            Sigma = S/Z - np.dot(mu, mu.T)
            sigmaTied += Z*Sigma
            gmmNew.append((w, mu))
        #get tied covariance
        gmm = gmmNew
        sigmaTied = sigmaTied/N
        U,s,_ = np.linalg.svd(sigmaTied)
        s[s<psi]=psi 
        sigmaTied = np.dot(U, mcol(s)*U.T)
        
        gmmNew=[]
        for g in range(len(gmm)):
            (w,mu)=gmm[g]
            gmmNew.append((w,mu,sigmaTied))
        gmm=gmmNew
        

    return gmm

def GMM_LBG(X, doub, version):
    assert version == 'FullCovariance' or version == 'DiagonalCovariance' or version == 'TiedCovariance',"GMM version not correct"
    init_mu, init_sigma = mean_and_covarianceMatrix(X)
    
    if version == "DiagonalCovariance" :
         init_sigma = init_sigma * np.eye(X.shape[0]) # We need an initial diagonal GMM to train a diagonal GMM
    
    psi = 0.01
    U, s, _ = np.linalg.svd(init_sigma)
    s[s<psi] = psi
    init_sigma = np.dot(U, mcol(s)*U.T)
    
    gmm = [(1.0, init_mu, init_sigma)]
    
    for i in range(doub):
        doubled_gmm = []
        
        for component in gmm: 
            w = component[0]
            mu = component[1]
            sigma = component[2]
            U, s, Vh = np.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * 0.1 # 0.1 is alpha
            component1 = (w/2, mu+d, sigma)
            component2 = (w/2, mu-d, sigma)
            doubled_gmm.append(component1)
            doubled_gmm.append(component2)
            if version == "FullCovariance" :
                gmm = GMM_EM(X, doubled_gmm)
            elif version == "DiagonalCovariance":
                gmm = GMM_EM_diag(X, doubled_gmm)
            elif version == "TiedCovariance":
                gmm = GMM_EM_tied(X, doubled_gmm)
            
    return gmm

