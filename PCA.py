# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:48:29 2024

@author: Davide
"""
import numpy as np
import matplotlib.pyplot as plt


def mcol(v):
    return v.reshape((v.size, 1))

    ## PCA implementation
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

def createP(C, m):
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    return P

def PCA_impl(D, m):
    DC = centerData(D)
    C = createCenteredCov(DC)
    P = createP(C, m)
    DP = np.dot(P.T, D)

    return DP,P

def PCA_plot(D):
    # Calcola i valori propri della matrice di covarianza
    DC = centerData(D)
    C = createCenteredCov(DC)
    eigenvalues, _ = np.linalg.eigh(C)
    
    # Ordina i valori propri in ordine decrescente
    eigenvalues = eigenvalues[::-1]
    
    # Calcola la varianza spiegata per ogni componente principale
    explained_variance = eigenvalues / np.sum(eigenvalues)

    # Creare un grafico della varianza spiegata
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative variance')
    plt.show()