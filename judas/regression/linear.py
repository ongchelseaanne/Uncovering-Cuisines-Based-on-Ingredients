import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from .utils import progressbar
from tqdm.autonotebook import tqdm

class TrainLinear():
    
    alpha = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    var = alpha
    varname = 'alpha'
    
    def __init__(self, X, y, reg, Number_trials, scaler=None): 
        score_train = []
        score_test = []
        weighted_coefs_seeds = []
        self.reg = reg
        
        with tqdm(total=Number_trials*len(self.alpha)) as pb:
            i_coef=[]
            for seed in range(Number_trials):
                training_accuracy = []  
                test_accuracy = []
                weighted_coefs=[]     
                a_feature_coef = {}
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed + 1}')
                
                if reg == 'linear':
                    self.var = 0
                    lr = LinearRegression().fit(X_train, y_train)
                    training_accuracy = [lr.score(X_train, y_train)]
                    test_accuracy = [lr.score(X_test, y_test)]
                    weighted_coefs = [lr.coef_[0]]
                    pb.update(1*len(self.alpha))

                elif reg in ['lasso', 'ridge']:
                    for alpha_run in self.alpha:
                        if reg == 'lasso':
                            lr = Lasso(alpha=alpha_run).fit(X_train, y_train)
                            a_feature_coef[alpha_run] = lr.coef_
                        elif reg == 'ridge':
                            lr = Ridge(alpha=alpha_run).fit(X_train, y_train)

                        training_accuracy.append(lr.score(X_train, y_train))
                        test_accuracy.append(lr.score(X_test, y_test))
            
                        coefs = lr.coef_[0] 
                        weighted_coefs.append(coefs) #append all the computed coefficients per trial
                        pb.update(1)
                    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
                weighted_coefs_seeds.append(weighted_coefs)
                i_coef.append(a_feature_coef)

        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)
        mean_coefs=np.mean(weighted_coefs_seeds, axis=0) #get the mean of the weighted coefficients over all the trials 
        self.coefl = i_coef
        top_weights = np.abs(mean_coefs)[np.argmax(self.score)]
        top_pred_feature_index = np.argmax(top_weights)
        self.top_predictor = X.columns[top_pred_feature_index]  
            
        return

    def result(self):
        if self.reg != 'linear':
            return ['{}'.format(self.reg.title()), '{:.2%}'.format(np.amax(self.score)), \
                'alpha = {0}'.format(self.alpha[np.argmax(self.score)]), self.top_predictor]
        return ['{}'.format(self.reg.title()), '{:.2%}'.format(np.amax(self.score)), \
                'NA', self.top_predictor]
