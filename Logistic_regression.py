# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize
#from sklearn.datasets import load_iris
from Evaluations import * 

def logreg_obj_wrap(DTR, LTR, l, pi=None):
    if pi is None:
        def logreg_obj(v):
            w, b = v[:-1], v[-1]
            loss = np.sum(np.logaddexp(0, -((2 * LTR - 1) * (np.dot(w.T, DTR) + b))))
            reg_coefficient = l / 2 * np.linalg.norm(w)**2
            return reg_coefficient + loss / DTR.shape[1]
    else: 
        def logreg_obj(v):
            w, b = v[:-1], v[-1]
            st = np.sum(np.logaddexp(0, -(2 * LTR[LTR == 1] - 1) * (np.dot(w.T, DTR[:, LTR == 1]) + b)))
            sf = np.sum(np.logaddexp(0, -(2 * LTR[LTR == 0] - 1) * (np.dot(w.T, DTR[:, LTR == 0]) + b)))
            reg_coefficient = l / 2 * np.linalg.norm(w)**2
            nt= DTR[:, LTR == 1].shape[1]
            nf =DTR[:, LTR == 0].shape[1]
            return reg_coefficient + (pi / nt) * st + ((1 - pi) / nf) * sf
    return logreg_obj


def logistic_regression(DTR, LTR, l, DTE, LTE, pi=None, cal=True):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, pi)
    minimizer = scipy.optimize.fmin_l_bfgs_b(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)

    w, b = minimizer[0][:-1], minimizer[0][-1]
    scores = np.dot(w.T, DTE) + b
    llr = scores
    
    if (cal) and pi is not None:
        prior_weighted = np.log(pi / (1 - pi))
        llr = scores - prior_weighted
        
    return llr

# Quadratic expansion

def vec_xxT(D):
    D = mcol(D)  
    return np.dot(D, D.T).reshape(D.size ** 2)  

def validate(scores, DTE, LTE):
    
    predicted_labels = scores > 0
    wrong_predictions = np.sum(predicted_labels != LTE)
    samples_number = DTE.shape[1]
    error_rate = (wrong_predictions / samples_number) * 100
    print(f"Error Rate: {error_rate:.2f}%")

def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTar = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wNon = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(mcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTar # Apply the weights to the loss computations
        loss[ZTR<0] *= wNon

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTar # Apply the weights to the gradient computations
        G[ZTR < 0] *= wNon
        
        GW = (mrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    #print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]
"""
if __name__ == '__main__':
    #D, L = load_iris_binary()
    #(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    nt = DTR[:, LTR == 1].shape[1]
    pt = nt / DTR.shape[1]
    l = 0.001  # Try different values here
   
    print(f"Testing with lambda = {l}")  # Debugging lambda
    scores = logistic_regression(DTR, LTR, l, DVAL, LVAL, pi=0.8, cal=True)
    
    #validate(scores, DVAL, LVAL)
    "For the non weighted application only"
    #scores -= np.log(pt / (1 - pt))
    
    DCF= compute_act_DCF(scores, LVAL, 0.5, 1, 1, th=None)
    min_DCF = compute_min_DCF(scores, LVAL, 0.5, 1, 1)
    print(f"DCF: {DCF:.3f}%")
    print(f"min_DCF: {min_DCF:.3f}%")
"""