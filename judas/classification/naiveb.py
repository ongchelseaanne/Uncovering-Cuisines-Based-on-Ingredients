import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from .utils import progressbar
from tqdm.autonotebook import tqdm
from imblearn.under_sampling import RandomUnderSampler

class TrainNB():
    
    alpha = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]
    var = alpha
    varname = 'alpha'

    def __init__(self, X, y, tp, Number_trials,alpha=None,scaler=None):
        if alpha is not None:
            self.alpha = alpha
            self.var = alpha
        score_train = []
        score_test = []
        self.type = tp
        
        with tqdm(total=Number_trials*len(self.alpha)) as pb:
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
                for alpha_run in self.alpha:
                    if tp == 'B':
                        lr=BernoulliNB(alpha=alpha_run).fit(X, y) 
                        a_feature_coef[alpha_run] = lr.coef_
                        coefs = lr.coef_[0] 
                    elif tp == 'G':
                        lr=GaussianNB().fit(X, y)
                    elif tp == 'M':
                        lr=MultinomialNB(alpha=alpha_run).fit(X, y)
                        a_feature_coef[alpha_run] = lr.coef_
                        coefs = lr.coef_[0] 
                    training_accuracy.append(lr.score(X_train, y_train))
                    test_accuracy.append(lr.score(X_test, y_test))

                    pb.update(1)
                    
                    
                score_train.append(training_accuracy)
                score_test.append(test_accuracy)
                i_coef.append(a_feature_coef)

        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)
        self.coefl = i_coef
        self.top_predictor = 'NA'    
            
        return

    def result(self):
        return ['Naive-Bayes ({0})'.format(self.type), '{:.2%}'.format(np.amax(self.score)), \
                'alpha = {0}'.format(self.alpha[np.argmax(self.score)]), self.top_predictor]
    
    def plot_max(self):
        return "NA"
