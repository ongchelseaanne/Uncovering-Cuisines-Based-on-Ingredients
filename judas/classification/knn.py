import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from .utils import progressbar
from tqdm.autonotebook import tqdm

class TrainKNN():

    neighbors_settings = range(1,70)    
    var = neighbors_settings
    varname = 'k neighbors'

    def __init__(self, X, y, Number_trials, neighbors_settings=None, scaler=None):
        score_train = []
        score_test = []
        if neighbors_settings is not None:
            self.neighbors_settings = neighbors_settings
            self.var = neighbors_settings
        
        with tqdm(total=Number_trials*len(self.neighbors_settings)) as pb:
            for seed in range(Number_trials):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                if scaler is not None:
                    scaler_inst = scaler.fit(X_train)
                    X_train = scaler_inst.transform(X_train)
                    X_test = scaler_inst.transform(X_test)
                pb.set_description(f'Trial: {seed + 1}')
            
                acc_train = []
                acc_test = []

                for n_neighbors in self.neighbors_settings:   
                    clf = KNeighborsClassifier(n_neighbors=n_neighbors) # build the model 
                    clf.fit(X_train, y_train)    
                    acc_train.append(clf.score(X_train, y_train))
                    acc_test.append(clf.score(X_test, y_test))
                    pb.update(1)

                score_train.append(acc_train)
                score_test.append(acc_test)   
            
        self.score = np.mean(score_test, axis=0)
        self.sc_train = np.mean(score_train, axis=0)
        self.std_score = np.std(score_test, axis=0)
        self.std_train = np.std(score_train, axis=0)

        return

    def result(self):
        return ['kNN', '{:.2%}'.format(np.amax(self.score)), 
                'N_Neighbor = {0}'.format(np.argmax(self.score)+1), 'NA']
