# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:57:28 2024

@author: Davide
"""
import numpy as np
import scipy.optimize


def mcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def mrow(x): # Same as in pca script
    return x.reshape((1, x.size))

def JDual(alpha,H):
        Ha = np.dot(H,mcol(alpha))
        aHa = np.dot(mrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
def LDual(alpha,H):
        loss, grad = JDual(alpha,H)
        return -loss, -grad

def JPrimal(w,D_ext,Z,C):
        S = np.dot(mrow(w), D_ext)
        loss = np.maximum(np.zeros(S.shape), 1-Z*S).sum()
        JPrim = 0.5 * np.linalg.norm(w)**2 + C * loss
        print(f"JPRIM : {JPrim}")
        return JPrim
    
def SVM_linear(DTR, LTR, DTE, C, K):
    # build the extended training data matrix D_ext
    D_ext = np.vstack((DTR, K * np.ones((1, DTR.shape[1]))))

    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = np.where(LTR == 1, 1, -1)

    # compute H_hat
    G = np.dot(D_ext.T,D_ext)
    H = mcol(Z) * mrow(Z) * G

    alphaStar , _ , _ = scipy.optimize.fmin_l_bfgs_b(
    lambda alpha: LDual(alpha, H),
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(D_ext, mcol(alphaStar) * mcol(Z))
    """    
    JPrim = JPrimal(wStar,D_ext,Z,C)
    loss_func,_= LDual(alphaStar,H)
    print(f"LDUAL : {loss_func.min()}")
    duality_gap = JPrim + loss_func.min()
    print(duality_gap)    
""" 
    k_row = np.ones((1, DTE.shape[1])) * K
    DTE_EXT = np.vstack((DTE, k_row))
    # make scores
    Scores = np.dot(wStar.T,DTE_EXT)
    
    return Scores.ravel()

def SVM_Poly(DTR, LTR, DTE, C, K, d, c):
    
    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = np.where(LTR == 1, 1, -1)
    # Compute H_hat directly on training data, no expansion needed. 
    # and compute H using kernel function instead of dot product
    H = ((np.dot(DTR.T,DTR) + c) ** d) + K * K
    H = mcol(Z) * mrow(Z) * H
     
    alphaStar , _x, _y = scipy.optimize.fmin_l_bfgs_b(
    lambda alpha: LDual(alpha, H),
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(DTR, mcol(alphaStar) * mcol(Z))
    
    kernel = ((np.dot(DTR.T,DTE) + c) ** d) + K * K #kernel based on product between TRAINING and TEST samples
    scores = np.sum(np.dot(alphaStar * mrow(Z), kernel), axis=0)
    return scores.ravel() # ravel to get a 1D array (N,) instead of a 2D (1,N) 

def SVM_RBF(DTR, LTR, DTE, C, K, gamma):
    
    # compute the zi for each train sample, z = 1 if class is 1, -1 if class is 0
    Z = np.where(LTR == 1, 1, -1)    
    # Compute H_hat directly on training data, no expansion needed. 
    # and compute H using kernel function instead of dot product
    H = np.zeros((DTR.shape[1], DTR.shape[1]))
    
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            H[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = mcol(Z) * mrow(Z) * H

    alphaStar , _x, _y = scipy.optimize.fmin_l_bfgs_b(
    lambda alpha: LDual(alpha, H),
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(DTR, mcol(alphaStar) * mcol(Z))
    
    # compute kernel on train and TEST set
    kernel = np.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernel[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K
    

    scores = np.sum(np.dot(alphaStar * mrow(Z), kernel), axis=0)
    return scores.ravel()