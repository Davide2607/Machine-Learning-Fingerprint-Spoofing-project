# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:10:47 2024

@author: Davide
"""


import numpy as np
import scipy as sp


def mcol(v):
    return v.reshape((v.size, 1))

def createCenteredCov(DC):
    C = 0
    for i in range(DC.shape[1]):
        C += np.dot(DC[:, i:i+1], DC[:, i:i+1].T)

    C /= float(DC.shape[1])
    return C

def centerData(D):
    mu = D.mean(1)
    DC = D - mcol(mu)
    return DC

def createSBSW(D,L):


    hLabels = {
        0: 'Fake fingerprint',
        1: 'Authentic fingerprint',
    }
    D0 = D[:, L == hLabels[0]]
    D1 = D[:, L == hLabels[1]]


    DC0 = centerData(D0)
    DC1 = centerData(D1)

    SW1 = createCenteredCov(DC0)
    SW2 = createCenteredCov(DC1)

    centeredSamples = [DC0,DC1]
    allSWc = [SW1,SW2]

    samples = [D0,D1]
    mu = mcol(D.mean(1))

    SB=0
    SW=0

    for x in range(2):
        m = mcol(samples[x].mean(1))
        SW = SW + (allSWc[x]*centeredSamples[x].shape[1])
        SB = SB + samples[x].shape[1] * np.dot((m-mu),(m-mu).T)    
                                                                   
    SB = SB/(float)(D.shape[1])
    SW = SW / (float)(D.shape[1])

    return SB,SW

def LDA(D,L,m):

    SB, SW = createSBSW(D,L)        
    s,U = sp.linalg.eigh(SB,SW) 
    W = U[:,::-1][:,0:m]        

    return W

def LDA_impl(D,W) :
    #W1 = LDA(D,L,m)
    DW = np.dot(W.T,D)     
    return DW

def split_db_2to1(D, L, seed=0):

  nTrain = int(D.shape[1]*2.0/3.0)
  np.random.seed(seed)
  idx = np.random.permutation(D.shape[1])
  idxTrain = idx[0:nTrain]
  idxTest = idx[nTrain:]
  DTR = D[:, idxTrain]
  DVAL = D[:, idxTest]
  LTR = L[idxTrain]
  LVAL = L[idxTest]
  return (DTR, LTR), (DVAL, LVAL)
