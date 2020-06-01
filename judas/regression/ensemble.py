import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from .utils import progressbar
from tqdm.autonotebook import tqdm

class TrainDecisionTree():

    maxdepth_settings = range(1, 20)
    var = maxdepth_settings
    varname = 'maxdepth'

    def __init__(self, X, y, Number_trials, maxdepth_settings=None,scaler=None):
        score_train = []
        score_test = []
        if maxdepth_settings is not None:
            self.maxdepth_settings = maxdepth_settings
            self.var = maxdepth_settings

        with tqdm(total=Number_trials*len(self.maxdepth_settings)) as pb:
            for seed in range(1,Number_trials+1,1):
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seed)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed}')
                training_accuracy = []
                test_accuracy = []

                for depth in maxdepth_settings:   
                    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)  # build the model
                    tree.fit(X_train, y_train)
    
                    training_accuracy.append(tree.score(X_train, y_train)) # record training set accuracy
                    test_accuracy.append(tree.score(X_test, y_test))   # record generalization accuracy
                    pb.update(1)
    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
                
        self.score = np.mean(score_test, axis=0) 
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)

        # get top predictor
        best_depth = maxdepth_settings[np.argmax(self.score)]
        tree = DecisionTreeRegressor(max_depth=best_depth, random_state=42)  # build the model
        tree.fit(X_train, y_train)
        self.top_predictor = X.columns[np.argmax(tree.feature_importances_)]
        
        abs_mean_coefs = np.abs(tree.feature_importances_)
        coefs_count = len(abs_mean_coefs)
        fig, ax = plt.subplots(figsize=(3,5))
        ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))
        ax.set_yticks(np.arange(coefs_count))
        ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])
        #self.top_predictor='NA'
        return

    def result(self):
        return ['Decision Trees', '{:.2%}'.format(np.amax(self.score)), \
                'depth = {0}'.format(self.maxdepth_settings[np.argmax(self.score)]), self.top_predictor]


class TrainRandomForest():
    # n_estimators_settings = range(1, 20) # try n_neighbors from 1 to 50
    n_estimators_settings = range(1, 20)
    var = n_estimators_settings
    varname = 'n_estimators'

    def __init__(self,X,y, Number_trials, n_estimators_settings=range(1,20),scaler=None):
        self.n_estimators_settings = n_estimators_settings
        score_train = []
        score_test = []
        if n_estimators_settings is not None:
            self.n_estimators_settings = n_estimators_settings
            self.var = n_estimators_settings
            
        with tqdm(total=Number_trials*len(self.n_estimators_settings)) as pb:    
            for seed in range(1,Number_trials+1,1):
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seed)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed}')
                training_accuracy = []
                test_accuracy = []
            
                for estimator in n_estimators_settings:   
                    forest = RandomForestRegressor(n_estimators=estimator, random_state=0, max_features='auto')
                    forest.fit(X_train, y_train)
                    training_accuracy.append(forest.score(X_train, y_train)) # record training set accuracy
                    test_accuracy.append(forest.score(X_test, y_test))   # record generalization accuracy
                    pb.update(1)

                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
        
        self.score = np.mean(score_test, axis=0) 
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)

        # get top predictor
        best_estimator = n_estimators_settings[np.argmax(self.score)]
        forest = RandomForestRegressor(n_estimators=best_estimator, random_state=0, max_features='auto')  # build the model
        forest.fit(X_train, y_train)
        self.top_predictor = X.columns[np.argmax(forest.feature_importances_)]
        
        abs_mean_coefs = np.abs(forest.feature_importances_)
        coefs_count = len(abs_mean_coefs)
        fig, ax = plt.subplots(figsize=(3,5))
        ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))
        ax.set_yticks(np.arange(coefs_count))
        ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])
        
        #self.top_predictor='NA'
        return

    def result(self):
        return ['Random Forest', '{:.2%}'.format(np.amax(self.score)), \
                'n-estimator = {0}'.format(self.n_estimators_settings[np.argmax(self.score)]), self.top_predictor]

class TrainGBM():

    maxdepth_settings = range(1, 10)
    var = maxdepth_settings
    varname = 'maxdepth'
    
    def __init__(self,X,y, Number_trials, maxdepth_settings=None,scaler=None):
        score_train = []
        score_test = []
        if maxdepth_settings is not None:
            self.maxdepth_settings = maxdepth_settings
            self.var = maxdepth_settings
            
        with tqdm(total=Number_trials*len(self.maxdepth_settings)) as pb:
            for seed in range(1,Number_trials+1,1):
                X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seed)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed}')
                training_accuracy = []
                test_accuracy = []

                for depth in maxdepth_settings:   

                    gbrt = GradientBoostingRegressor(max_depth=depth, learning_rate=0.01, random_state=0)  # build the model
                    gbrt.fit(X_train, y_train)

                    training_accuracy.append(gbrt.score(X_train, y_train)) # record training set accuracy
                    test_accuracy.append(gbrt.score(X_test, y_test))   # record generalization accuracy
                    pb.update(1)
                    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
                
        self.score = np.mean(score_test, axis=0) 
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)

        # get top predictor
        best_depth = maxdepth_settings[np.argmax(self.score)]
        gbrt = GradientBoostingRegressor(max_depth=best_depth, learning_rate=0.01, random_state=0)  # build the model
        gbrt.fit(X_train, y_train)
        self.top_predictor = X.columns[np.argmax(gbrt.feature_importances_)]
        
        abs_mean_coefs = np.abs(gbrt.feature_importances_)
        coefs_count = len(abs_mean_coefs)
        fig, ax = plt.subplots(figsize=(3,5))
        ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))
        ax.set_yticks(np.arange(coefs_count))
        ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])
        return
    
    def result(self):
        return ['Gradient Boosting Method', '{:.2%}'.format(np.amax(self.score)), \
                'depth = {0}'.format(self.maxdepth_settings[np.argmax(self.score)]), self.top_predictor]
    