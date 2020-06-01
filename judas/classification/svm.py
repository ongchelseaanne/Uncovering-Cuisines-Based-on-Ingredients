import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from .utils import progressbar
from tqdm.autonotebook import tqdm
from imblearn.under_sampling import RandomUnderSampler

class TrainSVM():

    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    var = C
    varname = 'C'

    def __init__(self, X, y, reg, Number_trials,C=None,scaler=None):
        if C is not None:
            self.C = C
            self.var = C    
        score_train = []
        score_test = []
        weighted_coefs_seeds = []
        self.reg = reg
        
        with tqdm(total=Number_trials*len(self.C)) as pb:
            for seed in range(Number_trials):
                training_accuracy = []  
                test_accuracy = []
                weighted_coefs = []
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                under_samp = RandomUnderSampler()
                X_train, y_train = under_samp.fit_sample(X_train, y_train)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed + 1}')
                for alpha_run in self.C:
                    if reg == 'l1':
                        svc = LinearSVC(C=alpha_run, penalty=reg, loss='squared_hinge', dual=False).fit(X_train, y_train)
                    if reg == 'l2':
                        svc = LinearSVC(C=alpha_run, penalty=reg).fit(X_train, y_train)
                    training_accuracy.append(svc.score(X_train, y_train))
                    test_accuracy.append(svc.score(X_test, y_test))
                
                    coefs = svc.coef_[0]
                    weighted_coefs.append(coefs)
                    pb.update(1)
                        
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
                weighted_coefs_seeds.append(weighted_coefs)

        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)
        mean_coefs=np.mean(weighted_coefs_seeds, axis=0) #get the mean of the weighted coefficients over all the trials 
        top_weights = np.abs(mean_coefs)[np.argmax(self.score)]
        top_pred_feature_index = np.argmax(top_weights)
        self.top_predictor = X.columns[top_pred_feature_index]        
            
        return

    def result(self):
        return ['Linear SVM ({0})'.format(self.reg), '{:.2%}'.format(np.amax(self.score)), \
                'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]

class TrainNSVM():
    
    gamma = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    var = gamma
    varname = 'gamma'

    def __init__(self, X, y, Number_trials,gamma=None,scaler=None):
        if gamma is not None:
            self.gamma = gamma
            self.var = gamma
        score_train = []
        score_test = []
        
        with tqdm(total=Number_trials*len(self.gamma)) as pb:
            for seed in range(Number_trials):
                training_accuracy = []  
                test_accuracy = []
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                under_samp = RandomUnderSampler()
                X_train, y_train = under_samp.fit_sample(X_train, y_train)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed + 1}')

                for gamma_run in self.gamma:
                    svm = SVC(kernel='rbf', gamma=gamma_run, C=10)
                    svc = svm.fit(X_train, y_train)
                    
                    training_accuracy.append(svc.score(X_train, y_train))
                    test_accuracy.append(svc.score(X_test, y_test))
                    pb.update(1)
                    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
    
        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)
        self.top_predictor ='NA'
        return

    def result(self):
        return ['Nonlinear SVM (RBF)', '{:.2%}'.format(np.amax(self.score)), \
                'gamma = {0}'.format(self.gamma[np.argmax(self.score)]), self.top_predictor]

class TrainNSVMPoly():
    
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    var = C
    varname = 'C'

    def __init__(self, X, y, Number_trials,C=None,scaler=None):
        if C is not None:
            self.C = C
            self.var = C
        score_train = []
        score_test = []
        
        with tqdm(total=Number_trials*len(self.C)) as pb:
            for seed in range(Number_trials):
                training_accuracy = []  
                test_accuracy = []
                weighted_coefs = []
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                under_samp = RandomUnderSampler()
                X_train, y_train = under_samp.fit_sample(X_train, y_train)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed + 1}')
                
                for C_run in self.C:
                    svm = SVC(kernel='poly',degree=2, coef0=1, C=C_run)
                    svc=svm.fit(X_train, y_train)
                    
                    training_accuracy.append(svc.score(X_train, y_train))
                    test_accuracy.append(svc.score(X_test, y_test))
                    pb.update(1)
                    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
    
        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)
        self.top_predictor ='NA'
        return

    def result(self):
        return ['Nonlinear SVM (Poly)', '{:.2%}'.format(np.amax(self.score)), \
                'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]