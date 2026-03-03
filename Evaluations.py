# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:28:34 2024

@author: Davide
"""
import numpy as np
import matplotlib.pyplot as plt


def mrow(col):
    return col.reshape((1,col.size))

def mcol(row):
    return row.reshape((row.size,1))

def confusion_matrix(predicted,labels):
  
    classes = np.unique(labels)
    confmat = np.zeros((len(classes), len(classes)))
    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((labels == classes[i]) & (predicted == classes[j]))
    return confmat.T.astype(int)


def OptimalBayesDecisions(pi,Cfn,Cfp,llr, th = None):
    # Compute bayes optimal decision based on passed parameters.
  
    if th == None:
        th = -np.log((pi*Cfn)/((1-pi)*Cfp))
    OptDecisions = np.array([llr > th])
    return np.int32(OptDecisions)


def binary_DCF_u(M, pi, Cfn, Cfp):
    """Compute the empirical bayes risk, assuming thath M is a confusion matrix of a binary case"""
    FNR = M[0,1]/(M[0,1] + M[1,1])
    FPR = M[1,0]/(M[0,0] + M[1,0])
    return pi*Cfn*FNR + (1-pi)*Cfp*FPR # bayes risk's formula for binary case 


def binary_DCF(M, pi, Cfn, Cfp):
    """Compute the normalized bayes risk, assuming thath M is a confusion matrix of a binary case"""
    empBayes = binary_DCF_u(M,pi,Cfn,Cfp) 
    B_dummy = np.array([pi*Cfn, (1-pi)*Cfp]).min()
    return empBayes / B_dummy   


def compute_act_DCF(llr, labels, pi, Cfn, Cfp, th=None):
    """Compute the actual DCF, which is basically the normalized bayes risk"""
    #compute opt bayes decisions
    pred = OptimalBayesDecisions(pi, Cfn, Cfp, llr, th=th)
    #compute confusion matrix
    cm = confusion_matrix(pred,labels)
    #compute DCF and return it
    return binary_DCF(cm, pi, Cfn, Cfp)


def compute_min_DCF(llr, labels, pi, Cfn, Cfp):
    """Compute the minDCF"""
    
    t = np.sort(llr)
    dcfList = []
    
    for _th in t:
        dcfList.append(compute_act_DCF(llr, labels, pi, Cfn, Cfp, th=_th))
    return np.array(dcfList).min()

def bayes_error_plot(llr,labels, title):
    # compute the p-tilde values
    effPriorLogOdds = np.linspace(-4, 4,21)
    
    # initialize DCF and minDCF vectors that will be plotted
    DCF = np.zeros(effPriorLogOdds.size)
    minDCF = np.zeros(effPriorLogOdds.size)
    
    # set Cfn and Cfp to 1
    Cfn = 1
    Cfp = 1
    
    #compute DCF and minDCF for each value of p-tilde considered (21 in total)
    for idx, p_tilde in enumerate(effPriorLogOdds):
        
        #compute prior π
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        
        # compute DCF
        DCF[idx] = compute_act_DCF(llr, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF[idx] = compute_min_DCF(llr, labels, pi_tilde, Cfn, Cfp)
            
    plt.plot(effPriorLogOdds, DCF, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='minDCF', color='b')
    plt.xlabel('prior log odds')
    plt.ylabel("DCF")
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.legend()
    plt.title(title)
    plt.show()



def bayes_error_plot_comparison(llr1,llr2,llr3,labels, title):
    # compute the p-tilde values
    effPriorLogOdds = np.linspace(-4, 4,25)
    
    # initialize DCF and minDCF vectors that will be plotted
    DCF1 = np.zeros(effPriorLogOdds.size)
    minDCF1 = np.zeros(effPriorLogOdds.size)
    DCF2 = np.zeros(effPriorLogOdds.size)
    minDCF2 = np.zeros(effPriorLogOdds.size)    
    DCF3 = np.zeros(effPriorLogOdds.size)
    minDCF3 = np.zeros(effPriorLogOdds.size) 
    # set Cfn and Cfp to 1
    Cfn = 1
    Cfp = 1
    
    #compute DCF and minDCF for each value of p-tilde considered (21 in total)
    for idx, p_tilde in enumerate(effPriorLogOdds):
        
        #compute prior π
        pi_tilde = 1 / (1 + np.exp(-p_tilde))
        
        # compute DCF
        DCF1[idx] = compute_act_DCF(llr1, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF1[idx] = compute_min_DCF(llr1, labels, pi_tilde, Cfn, Cfp)

        # compute DCF
        DCF2[idx] = compute_act_DCF(llr2, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF2[idx] = compute_min_DCF(llr2, labels, pi_tilde, Cfn, Cfp)
        # compute DCF
        DCF3[idx] = compute_act_DCF(llr3, labels, pi_tilde, Cfn, Cfp)
        
        # compute minDCF
        minDCF3[idx] = compute_min_DCF(llr3, labels, pi_tilde, Cfn, Cfp)
            
    plt.plot(effPriorLogOdds, DCF1, label='actDCF_QUAD_LOG_REG', color='r')
    plt.plot(effPriorLogOdds, minDCF1, label='minDCF_QUADRATIC_LOG_REG', color='blue')
    plt.plot(effPriorLogOdds, DCF2, label='DCF_SVM_RBF', color='g')
    plt.plot(effPriorLogOdds, minDCF2, label='minDCF_SVM_RBF', color='y')
    plt.plot(effPriorLogOdds, DCF3, label='DCF_GMM_DIAF', color='magenta')
    plt.plot(effPriorLogOdds, minDCF3, label='minDCF_GMM_DIAG', color='black')
    
    plt.xlabel('Prior Log-odds')
    plt.ylabel("DCF")
    plt.ylim([0, 1])
    plt.xlim([-4, 4])
    plt.legend()
    plt.title(title)
    plt.show()
    
    
def ROC_plot(llr, labels,title):
    
    thresholds = np.sort(llr)
    print(thresholds.size)
    FPR = []
    FNR = []
    
    for t in thresholds:
        # compute opt bayes decisions
        predictions = np.array([llr > t])
        # compute confusion matrix
        m = confusion_matrix(predictions,labels)
        # *** extract FNR and FPR for each considered t ***
        FPR.append(m[1,0]/(m[0,0] + m[1,0]))
        FNR.append(m[0,1]/(m[0,1] +m[1,1])) 
        
    #plot ROC curve
    TPR=1-np.array(FNR)
    plt.plot(FPR,TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()
       