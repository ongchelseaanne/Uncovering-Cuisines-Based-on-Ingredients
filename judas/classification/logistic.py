import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .utils import progressbar
from tqdm.autonotebook import tqdm
from imblearn.under_sampling import RandomUnderSampler

class TrainLogistic():
    
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
            i_coef=[]
            for seed in range(Number_trials):
                training_accuracy = []  
                test_accuracy = []
                weighted_coefs = []
                a_feature_coef = {}
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                under_samp = RandomUnderSampler()
                X_train, y_train = under_samp.fit_sample(X_train, y_train)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed + 1}')

                for alpha_run in self.C:
                    lr = LogisticRegression(C=alpha_run, penalty=reg).fit(X_train, y_train)
                    training_accuracy.append(lr.score(X_train, y_train))
                    test_accuracy.append(lr.score(X_test, y_test))
                
                    coefs = lr.coef_[0] 
                    weighted_coefs.append(coefs) #append all the computed coefficients per trial
                    pb.update(1)
                    a_feature_coef[alpha_run] = lr.coef_
                    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
                weighted_coefs_seeds.append(weighted_coefs)
                i_coef.append(a_feature_coef)

        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)
        self.coefl = i_coef
        mean_coefs=np.mean(weighted_coefs_seeds, axis=0) #get the mean of the weighted coefficients over all the trials 
        top_weights = np.abs(mean_coefs)[np.argmax(self.score)]
        top_pred_feature_index = np.argmax(top_weights)
        self.top_predictor = X.columns[top_pred_feature_index]        
            
        return

    def result(self):
        return ['Logistic ({0})'.format(self.reg), '{:.2%}'.format(np.amax(self.score)), \
                'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]
