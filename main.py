# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:52:24 2024

@author: Davide
"""


from PCA import *
from DataAnalysis import *
from LDA import *
from Gaussian_distribution import *
from MVG_model import *
from Evaluations import *
from prettytable import PrettyTable
from Logistic_regression import *
from SVM_linear_kernel import *
from GMM import *

import numpy as np


if __name__ == '__main__':
    
    D, L = load('./Data/trainData.txt')

    #------------------------------------------
    #FEAUTURE ANALYSIS
    #plotTot(D, L)
    #plotCross(D,L,6)
    #plotSingle(D,L,6)
    #stats(D,L)
    #Mean_Var(D,L)
    #--------------------------------------------
    # PCA
    #DP,P = PCA_impl(D, 6)
    #PCA_plot(D)
    #plotCross(DP,L,2)
    #plotSingle(DP,L,6)
    #--------------------------------------------
"""   
    # LDA
    W = LDA(D,L,1)
    DW = LDA_impl(D,W)
    plotSingle(DW,L,1)
    #Applico il metodo per dividere il train dataset in train e validatiom
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    W_lda = LDA(DTR,LTR,1)
    DTR_lda = LDA_impl(DTR,W_lda)
    if DTR_lda[0, LTR== 'Fake fingerprint'].mean() > DTR_lda[0, LTR== 'Authentic fingerprint'].mean():
        W_lda = -W_lda
        DTR_lda = LDA_impl(DTR, W_lda)
        
    DVAL_lda = np.dot(W_lda.T,DVAL)
    #Calcolo delle proiezioni medie delle classi
    
    # Calcolo del threshold
    threshold = (DTR_lda[0, LTR == 'Fake fingerprint'].mean() + DTR_lda[0, LTR == 'Authentic fingerprint'].mean()) / 2.0
    LVAL_numeric = np.where(LVAL == 'Authentic fingerprint', 0, 1)
    DVAL_lda_numeric = np.where(DVAL_lda[0] >= threshold, 0, 1)
    predictions_numeric = np.where(DVAL_lda_numeric == 0, 'Authentic fingerprint', 'Fake fingerprint')
    # Calcolo del tasso di errore
    print('Error rate LDA: %.2f%%' % ( (predictions_numeric != LVAL).sum() / float(LVAL.size) *100 ))
   
    
    #----------------------------------------------------
    # PCA + LDA
    DTR_pca ,P_pca = PCA_impl(DTR, 2)
    DVAL_pca = np.dot(P_pca.T,DVAL)
    W_lda = LDA(DTR_pca,LTR,1)
    DTR_lda = LDA_impl(DTR_pca , W_lda)

    if DTR_lda[0, LTR == 'Fake fingerprint'].mean() > DTR_lda[0, LTR== 'Authentic fingerprint'].mean():
        W_lda = -W_lda
        DTR_lda = LDA_impl(DTR_pca , W_lda) 
    
    DVAL_lda = np.dot(W_lda.T,DVAL_pca)
    threshold = (DTR_lda[0, LTR == 'Fake fingerprint'].mean() + DTR_lda[0, LTR == 'Authentic fingerprint'].mean()) / 2.0
    LVAL_numeric = np.where(LVAL == 'Authentic fingerprint', 1, 0)
    DVAL_lda_numeric = np.where((DVAL_lda[0]) >= threshold, 1, 0)
    predictions_numeric = np.where(DVAL_lda_numeric == 1, 'Authentic fingerprint', 'Fake fingerprint')
    # Calcolo del tasso di errore
    print('Error rate PCA + LDA: %.2f%%' % ( (predictions_numeric != LVAL).sum() / float(LVAL.size) *100 ))
 
    #------------------------------------------------------
    # MVG_density_distribution

for i in range(D.shape[0]):
            feature_x = D[i:i+1, L == 'Fake fingerprint']
            feature_y = D[i:i+1, L == 'Authentic fingerprint']  
            
            mu0 = mcol(feature_x.mean(1))
            DCls_cent = centerData(feature_x)      
            C0 = createCenteredCov(DCls_cent)
            
            mu1 = mcol(feature_y.mean(1))           
            DCls_cent = centerData(feature_y)      
            C1 = createCenteredCov(DCls_cent)            
            # Seleziona il colore in base alla classe
            #color = 'blue' if cls == 'Fake fingerprint' else 'red'
            
            # Effettua il plot della covarianza
            plt.xlabel('feature '+str(i))
            conf_plot(feature_x,feature_y, mu0, C0,mu1,C1)
"""

#------------------------------------------------------
    # MVG_Model
    
#(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
#prior = 1.0/2.0

# Limita alle prime 4 feature per i dati di addestramento
#DTR = DTR[2:4, :]
# Limita alle prime 4 feature per i dati di test
#DTE = DTE[2:4, :]
"""
hLabels = {
    0: 'Fake fingerprint',
    1: 'Authentic fingerprint'}


#DTR0 = D[:, L == hLabels[0]]
#DTR1 = D[:, L == hLabels[1]]
#plot_correlations(DTR0)
#plot_correlations(DTR1)

#predictions = MVG_classify(DTR, LTR, prior, DTE, LTE)
MVG_llr(DTR, LTR, DTE, LTE,prior)

predictions = NB_classify(DTR, LTR, prior, DTE, LTE)
NB_llr(DTR, LTR, DTE, LTE,prior)
predictions = TCG_classify(DTR, LTR, prior, DTE, LTE)
TCG_llr(DTR, LTR, DTE, LTE,prior)
   

    #------------------------------------------------------
    # PCA +  MVG_Model    
DP,P = PCA_impl(DTR, 6) 
DTE_pca = np.dot(P.T,DTE)
predictions = MVG_classify(DP, LTR, prior, DTE_pca, LTE)
MVG_llr(DP, LTR, DTE_pca, LTE,prior)
predictions = NB_classify(DP, LTR, prior, DTE_pca, LTE)
NB_llr(DP, LTR, DTE_pca, LTE,prior)
predictions = TCG_classify(DP, LTR, prior, DTE_pca, LTE)
TCG_llr(DP, LTR, DTE_pca, LTE,prior)
"""
#--------------------------------------------------------
    
    # MVG Evaluation    
    
#LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)

"""
PCA_components = [None, 6, 5, 4, 3, 2, 1]

Cfn = 1
Cfp = 1
pi_list = [0.1, 0.5, 0.9]

# Modelli e relative funzioni di likelihood ratio
models = [(MVG_llr, "MVG"),
          (NB_llr, "Naive Bayes"),
          (TCG_llr, "Tied Covariance")]

txt_output = ""

for model, m_string in models:
    result = PrettyTable()
    result.align = "c"
    result.field_names = ["PCA", "minDCF(pi=0.1)", "actDCF(pi=0.1)", "minDCF(pi=0.5)", "actDCF(pi=0.5)", "minDCF(pi=0.9)", "actDCF(pi=0.9)"]

    # Ripristina i dati originali prima di applicare PCA per ciascun modello
    original_DTR = DTR.copy()
    original_DTE = DTE.copy()

    for PCA_m in PCA_components:
        if PCA_m is not None:
            # Applica PCA
            DTR_pca, P = PCA_impl(original_DTR, PCA_m)
            DTE_pca = np.dot(P.T, original_DTE)
        else:
            DTR_pca = original_DTR
            DTE_pca = original_DTE

        llr = model(DTR_pca, LTR, DTE_pca, LTE, prior)
        minDCF = np.zeros(3)
        actDCF = np.zeros(3)

        for i, p in enumerate(pi_list):
            minDCF[i] = compute_min_DCF(llr, LTE, p, Cfn, Cfp)
            actDCF[i] = compute_act_DCF(llr, LTE, p, Cfn, Cfp)

        result.add_row([PCA_m, np.round(minDCF[0], 3), np.round(actDCF[0], 3), np.round(minDCF[1], 3), np.round(actDCF[1], 3), np.round(minDCF[2], 3), np.round(actDCF[2], 3)])

    txt_output = txt_output + "\n" + f"-{m_string}\n" + str(result)

with open('Results/Eval_MVG_results.txt', 'w') as file:
    file.write(str(txt_output))
"""
#-------------------------------------------

# Bayes error plot GAUSSIAN CLASSIFIERS
"""
DTR_pca,P = PCA_impl(DTR, 4) 
DTE_pca = np.dot(P.T,DTE)

scores_MVG = MVG_llr(DTR, LTR, DTE, LTE, prior)
scores_NB = NB_llr(DTR, LTR, DTE, LTE, prior)
scores_TCG = TCG_llr(DTR, LTR, DTE, LTE, prior)

#bayes_error_plot(scores_MVG, LTE, "Bayes error plot MVG")
#bayes_error_plot(scores_NB, LTE, "Bayes error plot Naive Bayes")
#bayes_error_plot(scores_TCG, LTE, "Bayes error plot Tied Covariance")

#scores_MVG_PCA = MVG_llr(DP, LTR, DTE_pca, LTE, prior)
#scores_NB_PCA = NB_llr(DP, LTR, DTE_pca, LTE, prior)
scores_TCG_PCA = TCG_llr(DTR_pca, LTR, DTE_pca, LTE, prior)


bayes_error_plot_comparison(scores_MVG, scores_NB,scores_TCG, LTE,"Bayes error plot comparison")
"""


#--------------------------------------------------------------------

# Logistic Regression
"""
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)


PCA_components = [3]
pi = 0.1
Cfn = 1
Cfp = 1

l_list = np.logspace(-4, 2, 13)
nt = DTR[:, LTR == 1].shape[1]
pt = nt / DTR.shape[1]

# build vec(xxT) Expanded space

def vec_xxT(D):
    D = mcol(D)  
    return np.dot(D, D.T).reshape(D.size ** 2)  

results = PrettyTable()
results.align = "c"
results.field_names = ["PCA","λ", "minDCF ", "act_DCF"]

fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set(xlabel='λ', ylabel='DCF', title='Logistic Regression prior-weighted with PCA = 3')

for PCA_m in PCA_components:
   
   minDCFs = []
   actDCFs = []
   
   for l in l_list:
        D, L = load('./Data/trainData.txt')
        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
        LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
        LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)
        
        
        if PCA_m is not None:
        
           DTR, P = PCA_impl(DTR, PCA_m)
           DTE = np.dot(P.T, DTE)
           

        if PCA_m != None:
               
            DTR, P = PCA_impl(DTR, PCA_m) 

            # features expansion on training dataset, after PCA
        DTR_XXT = np.apply_along_axis(vec_xxT, 0, DTR)
        DTR = np.vstack([DTR_XXT, DTR])

        if PCA_m != None:

                DTE = np.dot(P.T, DTE)

            # feature expansion on eval data, after PCA
        DTE_XXT = np.apply_along_axis(vec_xxT, 0, DTE)
        DTE = np.vstack([DTE_XXT, DTE])

        scores = logistic_regression(DTR, LTR, l, DTE, LTE, pi=0.1, cal=True) 
         #pi!= None only for prior-weighted
        
        #scores -= np.log(pt / (1 - pt))

        minDCF = compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(scores, LTE, pi, Cfn, Cfp)

        minDCFs.append(minDCF)
        actDCFs.append(actDCF)

        results.add_row([PCA_m,l, np.round(minDCF, 3), np.round(actDCF, 3)])
    
   ax.plot(l_list, minDCFs, label=f'minDCF PCA = {PCA_m}')
   ax.plot(l_list, actDCFs, '--', label=f'actDCF PCA = {PCA_m}') 

#ax.plot(l_list, minDCF, label='minDCF')
#ax.plot(l_list, actDCF, '--', label='actDCF')     
    
plt.legend()
#plt.savefig('Grafici/Quadratic_Logistic_Regression.png')
plt.show()

print(results)      



DTR_sampled = DTR[:, ::50]
LTR_sampled = LTR[::50]


PCA_components = [None]
pi = 0.1
Cfn = 1
Cfp = 1
l_list = np.logspace(-4, 2, 13)
nt = DTR_sampled[:, LTR_sampled == 1].shape[1]
pt = nt / DTR_sampled.shape[1]

results = PrettyTable()
results.align = "c"
results.field_names = ["PCA", "λ", "minDCF ", "act_DCF"]

fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set(xlabel='λ', ylabel='DCF', title='Logistic Regression_Reduced_Dataset')

for PCA_m in PCA_components:
    minDCFs = []
    actDCFs = []
    
    for l in l_list:
        D, L = load('./Data/trainData.txt')
        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
        LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
        LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

        # Apply sampling filter to training data
        DTR_sampled = DTR[:, ::50]
        LTR_sampled = LTR[::50]

        if PCA_m is not None:
            DTR_sampled, P = PCA_impl(DTR_sampled, PCA_m)
            DTE = np.dot(P.T, DTE)

        scores = logistic_regression(DTR_sampled, LTR_sampled, l, DTE, LTE, pi=None, cal=False)
        scores -= np.log(pt / (1 - pt))

        minDCF = compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(scores, LTE, pi, Cfn, Cfp)

        minDCFs.append(minDCF)
        actDCFs.append(actDCF)

        results.add_row([PCA_m, l, np.round(minDCF, 3), np.round(actDCF, 3)])
    
    ax.plot(l_list, minDCFs, label=f'minDCF PCA-{PCA_m}')
    ax.plot(l_list, actDCFs, '--', label=f'actDCF PCA-{PCA_m}') 

plt.legend()
plt.savefig('Grafici/Logistic_Regression_Reduced_Dataset.png')
plt.show()

#print(results)
"""
#---------------------------------------------------------------------
"""
# LINEAR SVM

(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

pi = 0.1
Cfn = 1
Cfp = 1
C_list = np.logspace(-5,0,11).tolist() 
K=1

results = PrettyTable()
results.align = "c"
results.field_names = ["C", "K", "minDCF ", "act_DCF"]

fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set(xlabel='C', ylabel='DCF', title='Linear SVM_center_data')

minDCFs = []
actDCFs = []

DTR = centerData(DTR)
DTE = centerData(DTE)

for C in C_list:
         
        scores = SVM_linear(DTR, LTR, DTE, C, K)
        minDCF = compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(scores, LTE, pi, Cfn, Cfp)

        minDCFs.append(minDCF)
        actDCFs.append(actDCF)

        results.add_row([C, K, np.round(minDCF, 3), np.round(actDCF, 3)])

ax.plot(C_list, minDCFs, label='minDCF')
ax.plot(C_list, actDCFs, '--', label='actDCF') 

plt.legend()
plt.savefig('Grafici/Linear_SVM_center_data.png')
plt.show()
print(results)
"""
#----------------------------------------------------------
"""
#  SVM POLY

(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

pi = 0.1
Cfn = 1
Cfp = 1

C_list = np.logspace(-5,0,11).tolist() 
K=0
d=2
c=1

results = PrettyTable()
results.align = "c"
results.field_names = ["C", "K", "minDCF ", "act_DCF"]

fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set(xlabel='C', ylabel='DCF', title='SVM_Poly')

minDCFs = []
actDCFs = []

for C in C_list:
         
        scores = SVM_Poly(DTR, LTR, DTE, C, K,d,c)
        minDCF = compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(scores, LTE, pi, Cfn, Cfp)

        minDCFs.append(minDCF)
        actDCFs.append(actDCF)

        results.add_row([C, K, np.round(minDCF, 3), np.round(actDCF, 3)])

ax.plot(C_list, minDCFs, label='minDCF')
ax.plot(C_list, actDCFs, '--', label='actDCF') 

plt.legend()
#plt.savefig('Grafici/SVM_Poly.png')
plt.show()
print(results)
"""
"""
#------------------------------------------------------

#  SVM RBF

(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

pi = 0.1
Cfn = 1
Cfp = 1

C_list = np.logspace(-3,2,11).tolist() 
gamma_list = [np.exp(n) for n in range(-1,-5,-1)]
K=1

results = PrettyTable()
results.align = "c"
results.field_names = ["gamma","C", "K", "minDCF ", "act_DCF"]

fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set(xlabel='C', ylabel='DCF', title='SVM_RBF')


for i,gamma in enumerate(gamma_list,start=1):
    
    minDCFs = []
    actDCFs = []
    
    for C in C_list:
         
        scores = SVM_RBF(DTR, LTR, DTE, C, K, gamma)
        minDCF = compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
        actDCF = compute_act_DCF(scores, LTE, pi, Cfn, Cfp)

        minDCFs.append(minDCF)
        actDCFs.append(actDCF)

        results.add_row([gamma,C, K, np.round(minDCF, 3), np.round(actDCF, 3)])

    ax.plot(C_list, minDCFs, label=f'minDCF gamma e^{-i}')
    ax.plot(C_list, actDCFs, '--', label=f'actDCF gamma e^{-i}') 

plt.legend()
plt.savefig('Grafici/SVM_RBF.png')
plt.show()
print(results)
"""

#-----------------------------------------------------------------
"""
# GMM

(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

pi = 0.1
Cfn = 1
Cfp = 1

double_list0 = [0,1,2,3,4,5]
double_list1 = [0,1,2,3,4,5]

version_target =["FullCovariance","DiagonalCovariance","TiedCovariance"]
    
results = PrettyTable()
results.align = "c"
results.field_names = ["Components fake class","Components authentic class","Model","minDCF", "actDCF"]
    
for version in version_target:
     for double1 in double_list1:
         
           for double0 in double_list0:
        
              DTR0=DTR[:,LTR==0]                                  
              gmm_class0=GMM_LBG(DTR0, double0, version)  
              _, SM0=logpdf_GMM(DTE,gmm_class0)                    
            
             # same for class 1
              DTR1=DTR[:,LTR==1]                                  
              gmm_class1= GMM_LBG(DTR1, double1, version)
              _, SM1=logpdf_GMM(DTE,gmm_class1)
             
             # compute scores
              scores = SM1 - SM0 
              
              minDCF = compute_min_DCF(scores, LTE, pi, Cfn, Cfp)
              actDCF = compute_act_DCF(scores, LTE, pi, Cfn, Cfp)
              results.add_row([2**double0,2**double1,version,np.round(minDCF,3),np.round(actDCF,3)])

print(results)
"""

# BAYES PLOT OF THE BEST THREE MODELS( QUADRATIC LOG. REG. , SVM RBF, GMM)

(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

nt = DTR[:, LTR == 1].shape[1]
pt = nt / DTR.shape[1]

DTR_XXT = np.apply_along_axis(vec_xxT, 0, DTR)
DTR = np.vstack([DTR_XXT, DTR])
DTE_XXT = np.apply_along_axis(vec_xxT, 0, DTE)
DTE = np.vstack([DTE_XXT, DTE])

scores_QUAD_LOG_REG = logistic_regression(DTR, LTR, 0.03162277660168379, DTE, LTE, pi=None, cal=False)
scores_QUAD_LOG_REG  -= np.log(pt / (1 - pt))

D, L = load('./Data/trainData.txt')
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)
scores_SVM = SVM_RBF(DTR, LTR, DTE, 31.622776601683793, 1, 0.1353352832366127)


D, L = load('./Data/trainData.txt')
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

LTE = np.where(LTE == 'Authentic fingerprint', 1, 0)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)
DTR0=DTR[:,LTR==0]                                  
gmm_class0=GMM_LBG(DTR0, 3, "DiagonalCovariance")  
_, SM0=logpdf_GMM(DTE,gmm_class0)                    
            
DTR1=DTR[:,LTR==1]                                  
gmm_class1= GMM_LBG(DTR1, 5, "DiagonalCovariance")
_, SM1=logpdf_GMM(DTE,gmm_class1)

scores_GMM = SM1 - SM0 
#bayes_error_plot_comparison(scores_QUAD_LOG_REG, scores_SVM,scores_GMM, LTE,"Bayes error plot comparison best models")
print("JUVE MERDA")

#----------------------------------------------------------
# KFOLD + calibration

# Funzione per estrarre folds di training e validation per K-fold cross-validation
def extract_train_val_folds_from_ary(X, idx, KFOLD):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

"""
# Definizione dei parametri
KFOLD = 5

pT_values = [0.1, 0.5, 0.9]  # Esempio di valori di prior per cui calcolare minDCF e actDCF

# Inizializza la tabella dei risultati
results = PrettyTable()
results.align = "c"
results.field_names = ["pT", "minDCF", "actDCF"]

calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette

# Itera su ogni valore di pT
for pT in pT_values:
    
    # K-fold cross-validation
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_QUAD_LOG_REG, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
    scores = np.hstack(calibrated_scores)
    labels = np.hstack(label)
    
    minDCF_value = compute_min_DCF(scores, labels, 0.1, 1.0, 1.0)
    actDCF_value = compute_act_DCF(scores, labels, 0.1, 1.0, 1.0)  # Assumendo LVAL qui per actDCF

    results.add_row([pT, np.round(minDCF_value, 3), np.round(actDCF_value, 3)])

# Stampa la tabella dei risultati
print(results)


#THE BEST CHOICE FOR QUAD_LOG_REG IS WITH PT=0.5



pT=0.5
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_QUAD_LOG_REG, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

bayes_error_plot(scores, labels, "QUADRATIC_LOGISTIC_REGRESSION")

#THE BEST CHOICE FOR SVM_RBF IS WITH PT=0.1

pT=0.1
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_SVM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

bayes_error_plot(scores, labels, "SVM_RBF")
"""
#THE BEST CHOICE FOR GMM_DIAG IS WITH PT=0.1
"""
pT=0.1
KFOLD = 5

# Inizializza la tabella dei risultati
results = PrettyTable()
results.align = "c"
results.field_names = ["pT", "minDCF", "actDCF"]

calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette

    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

bayes_error_plot(scores, labels, "GMM_DIAGONAL")

"""
"""
#-----------------------------------------------------------
   # Fusion 
# Definizione dei parametri
# Funzione per estrarre folds di training e validation per K-fold cross-validation
def extract_train_val_folds_from_ary(X, idx, KFOLD):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]
KFOLD = 5

pT_values = [0.1, 0.5, 0.9]  # Esempio di valori di prior per cui calcolare minDCF e actDCF

# Inizializza la tabella dei risultati
results = PrettyTable()
results.align = "c"
results.field_names = ["pT", "minDCF", "actDCF"]

# Itera su ogni valore di pT
for pT in pT_values:   
    
    fusedScores = [] # We will add to the list the scores computed for each fold
    fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
    
    # Train KFOLD times the fusion model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training        
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_QUAD_LOG_REG, foldIdx,KFOLD)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores_SVM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, foldIdx,KFOLD)
        # Build the training scores "feature" matrix
        SCAL = np.vstack([SCAL1, SCAL2,SCAL3])
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        # Build the validation scores "feature" matrix
        SVAL = np.vstack([SVAL1, SVAL2,SVAL3])
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ SVAL + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        fusedScores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        fusedLabels.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
    fusedScores = np.hstack(fusedScores)
    fusedLabels = np.hstack(fusedLabels)


    minDCF_value = compute_min_DCF(fusedScores, fusedLabels, 0.1, 1.0, 1.0)
    actDCF_value = compute_act_DCF(fusedScores, fusedLabels, 0.1, 1.0, 1.0)  # Assumendo LVAL qui per actDCF

    results.add_row([pT, np.round(minDCF_value, 3), np.round(actDCF_value, 3)])


print(results)

#FUSION  Pt = 0.1

pT=0.1
KFOLD=5
fusedScores = []  # Lista per salvare i punteggi calibrati
fusedLabels = []  # Lista per salvare le etichette

for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training        
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_QUAD_LOG_REG, foldIdx,KFOLD)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores_SVM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LTE, foldIdx,KFOLD)
        # Build the training scores "feature" matrix
        SCAL = np.vstack([SCAL1, SCAL2,SCAL3])
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        # Build the validation scores "feature" matrix
        SVAL = np.vstack([SVAL1, SVAL2,SVAL3])
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ SVAL + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        fusedScores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        fusedLabels.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
fusedScores = np.hstack(fusedScores)
fusedLabels = np.hstack(fusedLabels)

bayes_error_plot(fusedScores, fusedLabels, "FUSION")





#------------------------------------------------------------------
# EVALUATION(eval_dataset)


DEVAL,LEVAL = load('./Data/evalData.txt')
LEVAL = np.where(LEVAL == 'Authentic fingerprint', 1, 0)
D, L = load('./Data/trainData.txt')
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

nt = DTR[:, LTR == 1].shape[1]
pt = nt / DTR.shape[1]

DTR_XXT = np.apply_along_axis(vec_xxT, 0, DTR)
DTR = np.vstack([DTR_XXT, DTR])
DEVAL_XXT = np.apply_along_axis(vec_xxT, 0, DEVAL)
DEVAL = np.vstack([DEVAL_XXT, DEVAL])

scores_QUAD_LOG_REG = logistic_regression(DTR, LTR, 0.03162277660168379, DEVAL, LEVAL, pi=None, cal=False)
scores_QUAD_LOG_REG  -= np.log(pt / (1 - pt))


DEVAL,LEVAL = load('./Data/evalData.txt')
LEVAL = np.where(LEVAL == 'Authentic fingerprint', 1, 0)
D, L = load('./Data/trainData.txt')
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)
scores_SVM = SVM_RBF(DTR, LTR, DEVAL, 31.622776601683793, 1, 0.1353352832366127)


DEVAL,LEVAL = load('./Data/evalData.txt')
LEVAL = np.where(LEVAL == 'Authentic fingerprint', 1, 0)
D, L = load('./Data/trainData.txt')
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)
DTR0=DTR[:,LTR==0]                                  
gmm_class0=GMM_LBG(DTR0, 3, "DiagonalCovariance")  
_, SM0=logpdf_GMM(DEVAL,gmm_class0)                    
            
DTR1=DTR[:,LTR==1]                                  
gmm_class1= GMM_LBG(DTR1, 5, "DiagonalCovariance")
_, SM1=logpdf_GMM(DEVAL,gmm_class1)

scores_GMM = SM1 - SM0 

"""
#-----------------------------------------------------------------------------
#EVALUATION(BEST DELIVERD SYSTEM GMM pi=0.1)
"""
# Definizione dei parametri
KFOLD = 5

pT_values = [0.1, 0.5, 0.9]  # Esempio di valori di prior per cui calcolare minDCF e actDCF

# Inizializza la tabella dei risultati
results = PrettyTable()
results.align = "c"
results.field_names = ["pT", "minDCF", "actDCF"]

calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette

# Itera su ogni valore di pT
for pT in pT_values:
    
    # K-fold cross-validation
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
    scores = np.hstack(calibrated_scores)
    labels = np.hstack(label)
    
    minDCF_value = compute_min_DCF(scores, labels, 0.1, 1.0, 1.0)
    actDCF_value = compute_act_DCF(scores, labels, 0.1, 1.0, 1.0)  # Assumendo LVAL qui per actDCF

    results.add_row([pT, np.round(minDCF_value, 3), np.round(actDCF_value, 3)])

# Stampa la tabella dei risultati
print(results)


#EVALUATIONGMM_DIAG WITH PT=0.1(TARGET APPLICATION)
KFOLD = 5
pT=0.1
calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)
minDCF = compute_min_DCF(scores, labels, 0.1, Cfn=1, Cfp=1)
actDCF= compute_act_DCF(scores, labels, 0.1, Cfn=1, Cfp=1)
print(f"GMM MINDCF(PT=0.1): {minDCF}")
print(f"GMM ACTDCF(PT=0.1): {actDCF}")
#bayes_error_plot(scores, labels, "EVALUATION GMM_DIAGONAL")

#EVALUATION GMM_DIAG  WITH PT=0.5

pT=0.5
calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

bayes_error_plot(scores, labels, "EVALUATION GMM_DIAGONAL (pt=0.5)")

#EVALUTION GMM_DIAG  WITH PT=0.9

pT=0.9
calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

bayes_error_plot(scores, labels, "EVALUATION GMM_DIAGONAL (pt=0.9)")


# EVALUATION SVM_RBF pt=0.1

pT=0.1
calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_SVM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

minDCF = compute_min_DCF(scores, labels, 0.1, Cfn=1, Cfp=1)
actDCF= compute_act_DCF(scores, labels, 0.1, Cfn=1, Cfp=1)
print(f"SVM RBF MINDCF: {minDCF}")
print(f"SVM RBF ACTDCF: {actDCF}")
bayes_error_plot(scores, labels, "EVALUATION SVM_RBF")

# EVALUATION Quadratic_logistic_regression pt=0.1

pT=0.1
calibrated_scores = []  # Lista per salvare i punteggi calibrati
label = []  # Lista per salvare le etichette
    # K-fold cross-validation
for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_QUAD_LOG_REG, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(mrow(SCAL), LCAL, 0,pT)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ mrow(SVAL) + b - np.log(pT / (1-pT))).ravel()

        # Aggiungi i punteggi calibrati e le etichette corrispondenti
        calibrated_scores.append(calibrated_SVAL)
        label.append(LVAL)
    
    # Calcola minDCF e actDCF per la fold di validation (ultima fold)
scores = np.hstack(calibrated_scores)
labels = np.hstack(label)

mminDCF = compute_min_DCF(scores, labels, 0.1, Cfn=1, Cfp=1)
actDCF= compute_act_DCF(scores, labels, 0.1, Cfn=1, Cfp=1)
print(f"QUAD_LOG_REG MINDCF: {minDCF}")
print(f"QUAD_LOG_REG ACTDCF: {actDCF}")
bayes_error_plot(scores, labels, "EVALUATION QUADRATIC_LOGISTIC-REGRESSION")


#eEVALUATION FUSION  Pt = 0.1


pT=0.1
KFOLD=5
fusedScores = []  # Lista per salvare i punteggi calibrati
fusedLabels = []  # Lista per salvare le etichette

for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training        
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_QUAD_LOG_REG, foldIdx,KFOLD)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_GMM, foldIdx,KFOLD)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores_SVM, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(LEVAL, foldIdx,KFOLD)
        # Build the training scores "feature" matrix
        SCAL = np.vstack([SCAL1, SCAL2,SCAL3])
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        # Build the validation scores "feature" matrix
        SVAL = np.vstack([SVAL1, SVAL2,SVAL3])
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ SVAL + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        fusedScores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        fusedLabels.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
fusedScores = np.hstack(fusedScores)
fusedLabels = np.hstack(fusedLabels)

minDCF = compute_min_DCF(fusedScores, fusedLabels, 0.1, Cfn=1, Cfp=1)
actDCF= compute_act_DCF(fusedScores, fusedLabels, 0.1, Cfn=1, Cfp=1)
print(f"FUSION MINDCF: {minDCF}")
print(f"FUSION ACTDCF: {actDCF}")
bayes_error_plot(fusedScores, fusedLabels, "EVALUATION FUSION")

#-------------------------------------------------------------------------
#EVALUATION GMM(ALL PARAMS AND COMPONENTS) ON THE EVALUATION SET

DEVAL,LEVAL = load('./Data/evalData.txt')
LEVAL = np.where(LEVAL == 'Authentic fingerprint', 1, 0)
D, L = load('./Data/trainData.txt')
(DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
LTR = np.where(LTR == 'Authentic fingerprint', 1, 0)

pi = 0.1
Cfn = 1
Cfp = 1

double_list0 = [0,1,2,3,4,5]
double_list1 = [0,1,2,3,4,5]

version_target =["FullCovariance","DiagonalCovariance","TiedCovariance"]
    
results = PrettyTable()
results.align = "c"
results.field_names = ["Components fake class","Components authentic class","Model","minDCF", "actDCF"]
    
for version in version_target:
     for double1 in double_list1:
         
           for double0 in double_list0:
        
              DTR0=DTR[:,LTR==0]                                  
              gmm_class0=GMM_LBG(DTR0, double0, version)  
              _, SM0=logpdf_GMM(DEVAL,gmm_class0)                    
            
             # same for class 1
              DTR1=DTR[:,LTR==1]                                  
              gmm_class1= GMM_LBG(DTR1, double1, version)
              _, SM1=logpdf_GMM(DEVAL,gmm_class1)
             
             # compute scores
              scores = SM1 - SM0 
              
              minDCF = compute_min_DCF(scores, LEVAL, pi, Cfn, Cfp)
              actDCF = compute_act_DCF(scores, LEVAL, pi, Cfn, Cfp)
              results.add_row([2**double0,2**double1,version,np.round(minDCF,3),np.round(actDCF,3)])

print(results)
"""