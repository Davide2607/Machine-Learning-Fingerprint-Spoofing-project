# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:58:25 2024

@author: Davide
"""

import numpy as np
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def load(file_path):

    # Inizializza le liste per le caratteristiche e le etichette di classe
    DList = []
    labelsList = []
    hLabels = {
        0: 'Fake fingerprint',
        1: 'Authentic fingerprint',
    }
    with open(file_path, 'r') as f:
    # Itera su ogni riga del file
      for line in f:
        # Dividi la riga in base alle virgole
        values = line.strip().split(',')

        # Estrai le caratteristiche dai primi valori della riga
        attrs = mcol(np.array([float(val.strip()) for val in values[:-1]]))

        # Estrai l'etichetta di classe dall'ultimo valore della riga
        name = values[-1].strip()

        # Controlla se l'etichetta di classe è vuota o mancante
        if name:
           label = hLabels[int(name)]
            # Aggiungi le caratteristiche e l'etichetta di classe alle rispettive liste
           DList.append(attrs)
           labelsList.append(label)

    # Converti le liste in array numpy
    D = np.hstack(DList)
    labels = np.array(labelsList)

    # Restituisci le caratteristiche e le etichette di classe
    return D, labels


def plotTot(D, L):
    hLabels = {
    0: 'Fake fingerprint',
    1: 'Authentic fingerprint',
    }
    D0 = D[:, L == hLabels[0]]
    D1 = D[:, L == hLabels[1]]

    plt.figure()
    plt.xlabel("Feature")
    plt.hist(D0[:, :], density=True, alpha=0.7, label=hLabels[0])
    plt.hist(D1[:, :], density=True, alpha=0.7, label=hLabels[1])
    plt.legend()
    plt.show()

def plotCross(D, L,m):

    hLabels = {
        0: 'Fake fingerprint',
        1: 'Authentic fingerprint',
    }
    D0 = D[:, L == hLabels[0]]
    D1 = D[:, L == hLabels[1]]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            plt.figure()
            plt.xlabel("Feature " + str(i))
            plt.ylabel("Feature " + str(j))
            plt.scatter(D0[i, :], D0[j, :], alpha = 0.5, label="Fake fingerprint")
            plt.scatter(D1[i, :], D1[j, :], alpha = 0.5,  label="Authentic fingerprint")
            plt.legend()
            plt.show()

def plotSingle(D, L,m):
    hLabels = {
        0: 'Fake fingerprint',
        1: 'Authentic fingerprint',
    }
    D0 = D[:, L == hLabels[0]]
    D1 = D[:, L == hLabels[1]]

    for i in range(m):
        plt.figure()
        plt.xlabel("Feature " + str(i))
        plt.ylabel("Number of elements")
        plt.hist(D0[i, :], bins=60,density=True, alpha=0.7, label="Fake fingerprint")
        plt.hist(D1[i, :], bins=60, density=True, alpha=0.7, label="Authentic fingerprint")
        plt.legend()
        plt.show()



def Mean_Var(D,L):

    for cls in ['Fake fingerprint','Authentic fingerprint']:
        print('Class', cls)
        DCls = D[:, L==cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu)

        var = DCls.var(1).reshape(D.shape[0], 1)
        std = DCls.std(1).reshape(D.shape[0], 1)
        print('Variance:\n', var)
        print('Std. dev.:\n', std)
        print()
        

def stats(D, L):

    mu = D.mean(1).reshape((D.shape[0], 1))
    print('Mean:')
    print(mu)
    print()
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    print('Covariance:')
    print(C)
    print()

    var = D.var(1).reshape((D.shape[0], 1))
    std = D.std(1).reshape((D.shape[0], 1))
    print('Variance:\n', var)
    print()
    print('Std. dev:\n', std)
    print()
    
 
     
def compute_correlation(X, Y):
    
    x_sum = np.sum(X)
    y_sum = np.sum(Y)

    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)

    sum_cross_prod = np.sum(X * Y)

    n = X.shape[0]
    
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr

def plot_correlations(DTR):
    corr = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            X = DTR[i, :]
            Y = DTR[j, :]
            pearson_elem = compute_correlation(X, Y)
            corr[i][j] = pearson_elem
    print(corr)