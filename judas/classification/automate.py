import time
from datetime import timedelta
import pandas as pd
import numpy as np
import warnings
import math

from .knn import TrainKNN
from .logistic import TrainLogistic
from .svm import TrainSVM, TrainNSVM, TrainNSVMPoly
from .ensemble import TrainDecisionTree, TrainRandomForest, TrainGBM
from .naiveb import TrainNB

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

class Judas():
    results = []
    models = []

    def __init__(self, DEBUG=False):
        self.DEBUG=DEBUG
        return

    def automate(self, X, y, models):
        warnings.filterwarnings("ignore")
        self.models = []
        for model in models:
            k = [k for k in model.keys()]
            if 'scaler' in k:
                scaler = model['scaler']
            else:
                scaler = None
            if 'trials' not in k:
                print('Number of Trials required')
                break;
            if model['model'] == 'knn':
                print('{}, n neighbors={}'.format(model['model'], model['k']))
                n = model['k']
                m = TrainKNN(X,y,Number_trials=model['trials'], neighbors_settings = n, scaler = scaler)
            elif model['model'] == 'logistic':
                print('{}, reg={}'.format(model['model'], model['reg']))                
                m = TrainLogistic(X,y, reg=model['reg'], Number_trials=model['trials'], scaler = scaler)
            elif model['model'] == 'svm':
                print('{}, reg={}'.format(model['model'], model['reg']))                
                m = TrainSVM(X,y, reg=model['reg'], Number_trials=model['trials'], scaler = scaler)
            elif model['model'] == 'nsvm-rbf':
                print('{}'.format(model['model']))                
                m = TrainNSVM(X,y, Number_trials=model['trials'], scaler = scaler)
            elif model['model'] == 'nsvm-poly':
                print('{}'.format(model['model']))                
                m = TrainNSVMPoly(X,y, Number_trials=model['trials'], scaler = scaler)
            elif model['model'] == 'ensemble-decisiontree':
                print('{}, max depth={}'.format(model['model'], model['maxdepth']))                
                m = TrainDecisionTree(X,y,Number_trials=model['trials'], maxdepth_settings=model['maxdepth'], scaler = scaler)
            elif model['model'] == 'ensemble-randomforest':
                print('{}, n estimators={}'.format(model['model'], model['n_est']))                
                m = TrainRandomForest(X,y,Number_trials=model['trials'], n_estimators_settings=model['n_est'], scaler = scaler)
            elif model['model'] == 'ensemble-gbm':
                print('{}, max depth={}'.format(model['model'], model['maxdepth']))                
                m = TrainGBM(X,y,Number_trials=model['trials'], maxdepth_settings=model['maxdepth'], scaler = scaler)
            elif model['model'] == 'naive-bayes':
                print('{}'.format(model['model']))                
                m = TrainNB(X,y,Number_trials=model['trials'], tp=model['tp'],scaler = scaler)
            else:
                continue
            self.models.append(m)

    def score(self):
        cols = ['Machine Learning Method', 'Test Accuracy', 'Best Parameter', 'Top Predictor Variable']
        df = pd.DataFrame(columns=cols)
        #print(self.DEBUG)
        for idx, m in enumerate(self.models):
            if self.DEBUG == True:
                print(idx)
            df.loc[idx] = m.result()
        return df
   
    def plot_accuracy(self,ax=None):
        def model_plot(model,ax):
            ax.plot(model.var, model.sc_train,
                    label="training accuracy")
            ax.plot(model.var, model.score, label="test accuracy")
            ax.fill_between(model.var,
                            model.sc_train-model.std_train,
                            model.sc_train+model.std_train, alpha=0.2)
            ax.fill_between(model.var, 
                            model.score-model.std_score,
                            model.score+model.std_score, alpha=0.2)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel(model.varname)
            ax.legend()
            return ax
        k = [mod for mod in self.models]
        klen = len(k)
        row = int(np.round(np.sqrt(klen),0))
        col = math.ceil(klen/row)
        fig, axes = plt.subplots(row,col, figsize=(15, row*5))
        i = 0
        if col == 1 and row == 1:
            ax = model_plot(k[i],axes)
            ax.set_title(k[i].result()[0])
            if k[i] != 'KNN':
                ax.set_xscale('log')
        elif col == 2 and row == 1:
            for ax in axes:
                ax = model_plot(k[i],ax)
                ax.set_title(k[i].result()[0])
                if k[i] != 'KNN':
                    ax.set_xscale('log')
                i+=1
        else:
            for axrow in axes:
                for ax in axrow:
                    if i >= len(k):
                        ax.axis("off")
                    else:
                        ax = model_plot(k[i],ax)
                        ax.set_title(k[i].result()[0])
                        if k[i] != 'KNN':
                            ax.set_xscale('log')
                    i+=1