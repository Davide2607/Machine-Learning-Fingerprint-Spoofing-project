# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:25:15 2024

@author: Davide
"""

#from FeautureAnalysis.PCA import *
from Gaussian_distribution import *

import numpy as np
import scipy as sp


def MVG_model(DTR,LTR):
 
    means = []
    Cov_matrices = []
    
    hLabels = {
        0: 'Fake fingerprint',
        1: 'Authentic fingerprint'
    }

    D1 = DTR[:, LTR == hLabels[0]]
    D2 = DTR[:, LTR == hLabels[1]]

    D1_cent = centerData(D1)
    D2_cent = centerData(D2)

    Cov_matrices.append(createCenteredCov(D1_cent))        
    Cov_matrices.append(createCenteredCov(D2_cent))
    
    means.append(mcol(D1.mean(1)))
    means.append(mcol(D2.mean(1)))

    return means,Cov_matrices,(D1.shape[1],D2.shape[1])


def TCG_model(DTR,LTR):

    S_matrix = 0
    means,Cov_matrices,Dn = MVG_model(DTR, LTR)
    
    Dn = np.array(Dn)
    
    Cov_matrices = np.array(Cov_matrices)
      
    for i in range(Dn.shape[0]):
        
        S_matrix += Dn[i]*Cov_matrices[i]  
    
    S_matrix /=DTR.shape[1]
    
    return means,S_matrix


def loglikelihoods(DTE,LTE,means,Cov_matrices,prior):
    
    class_likelihoods=[]
    labelset = np.unique(LTE,return_counts=True)
    num_Label = len(labelset)
    for i in range(num_Label):
        mu=means[i]
        c= Cov_matrices[i]
        ll=logpdf_GAU_ND(DTE, mu, c)
        class_likelihoods.append(ll)

    return np.squeeze(np.array(class_likelihoods))
    

def validate(predicted_labels,DTE,LTE):

    LTE_numeric = np.where(LTE == 'Authentic fingerprint', 1, 0)
    wrongPredictions = (LTE_numeric != predicted_labels).sum()
    samplesNumber = DTE.shape[1]
    errorRate = float(wrongPredictions / samplesNumber * 100)
    print(errorRate)

def pred_llr(LTE,llr):
  
    predictions_numeric = np.where(llr > 0.0, 'Authentic fingerprint', 'Fake fingerprint')

   # Stampa le previsioni e calcola il tasso di errore
    print('Labels:     ', LTE)
    print('Predictions:', predictions_numeric)
    print('Error rate: %.15f%%' % ( (predictions_numeric != LTE).sum() / float(LTE.size) *100 ))
  
    
def log_post_prob(log_SJoint):
        
    log_SMarginal = mrow(sp.special.logsumexp(log_SJoint,axis=0))

    log_SPost = log_SJoint - log_SMarginal
    
    scores = np.exp(log_SPost)
    predictedLabels = scores.argmax(axis=0)
    
    return predictedLabels
        
    
def MVG_classify(DTR,LTR,prior,DTE,LTE):
    
    means,Cov_matrices,_ = MVG_model(DTR,LTR) #3 means and 3 S_matrices -> 1 for each class (3 classes)
    
    logScores = loglikelihoods(DTE,LTE,means,Cov_matrices,prior)
    
    log_Sjoint = logScores + np.log(prior)

    predictedLabels = log_post_prob(log_Sjoint)
    #validate(predictedLabels,DTE,LTE)
    return predictedLabels


def MVG_llr(DTR,LTR,DTE,LTE,prior):


    means,Cov_matrices,_ = MVG_model(DTR,LTR) 

    logScores= loglikelihoods(DTE,LTE,means,Cov_matrices,prior)
    llr = logScores[1]-logScores[0]
    #pred_llr(LTE,llr)
    
    return llr

def NB_classify(DTR,LTR,prior,DTE,LTE):
     
  means,Cov_matrices,_ = MVG_model(DTR,LTR) 
  labelset = np.unique(LTE,return_counts=True)
  num_Label = len(labelset)
  for i in range(num_Label):
      Cov_matrices[i] = Cov_matrices[i]*np.eye(Cov_matrices[i].shape[0])
    
  logScores = loglikelihoods(DTE,LTE,means,Cov_matrices,prior)

  log_Sjoint = logScores + np.log(prior)

  predictedLabels = log_post_prob(log_Sjoint)
  #validate(predictedLabels,DTE,LTE)
  return predictedLabels


def NB_llr(DTR,LTR,DTE,LTE,prior):
    
    means,Cov_matrices,_ = MVG_model(DTR,LTR) 
    labelset = np.unique(LTE,return_counts=True)
    num_Label = len(labelset)
    for i in range(num_Label):
         Cov_matrices[i] = Cov_matrices[i]*np.eye(Cov_matrices[i].shape[0]) 
    
    logScores= loglikelihoods(DTE,LTE,means,Cov_matrices,prior)
    llr = logScores[1]-logScores[0]
    #pred_llr(LTE,llr)
    return llr
 

def TCG_classify(DTR,LTR,prior,DTE,LTE):    
    
    means,Cov_matrix = TCG_model(DTR,LTR)

    labelset = np.unique(LTE,return_counts=True)
    num_Label = len(labelset)
    Cov_matrices = np.empty(num_Label,dtype=object)
    for i in range(num_Label):
        Cov_matrices[i]= Cov_matrix
    
    logScores = loglikelihoods(DTE,LTE,means,Cov_matrices,prior)
    
    log_Sjoint = logScores + np.log(prior)

    predictedLabels = log_post_prob(log_Sjoint)
    #validate(predictedLabels,DTE,LTE)
    
    return predictedLabels

def TCG_llr(DTR,LTR,DTE,LTE,prior):
    
    means,Cov_matrix = TCG_model(DTR,LTR)
    
    Cov_matrices=[]
    labelset = np.unique(LTE,return_counts=True)
    num_Label = len(labelset)
    Cov_matrices = np.empty(num_Label,dtype=object)
    for i in range(num_Label):
        Cov_matrices[i] = Cov_matrix
        
    logScores= loglikelihoods(DTE,LTE,means,Cov_matrices,prior)
    llr = logScores[1]-logScores[0]
    #pred_llr(LTE,llr)
    return llr