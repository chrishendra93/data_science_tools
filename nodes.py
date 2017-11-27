import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


class Nodes(object):

    ''' A node is a representation of a random variable together with its pars'''

    def __init__(self, X, feature_name, feature_type='c', par_features=[], par_types=[]):

        '''par_types are either empty, c (continuous) or d (discrete)'''
        ''' X are training data in the form of numpy array or data frame'''
        assert len(par_features) == len(par_types), "lengths of pars and types do not match"
        assert set(par_types).issubset(set(['c', 'd'])), "invalid data type"

        if len(par_features) == 0:
            assert isinstance(X, pd.Series) or isinstance(X, pd.DataFrame), "expecting a series or a dataframe"
        else:
            assert isinstance(X, pd.Series) or isinstance(X, pd.DataFrame), "expecting a dataframe"

        self.training_df = X
        self.par_features = tuple(par_features)
        self.par_types = tuple(par_types)
        self.continuous_pars = []
        self.discrete_pars = []
        self.features = []
        self.dims = []
        self.joint_cpt = []
        self.joint_kde = {}
        self.par_cpt = []
        self.par_kde = {}

    def var_reorder(self, vars):
        return np.sort(vars)

    def fit(self):
        ''' P(X|pars_X) = P(X, pars_X) / P(pars_X)'''
        for i in range(len(self.par_features)):
            if self.par_types[i] == 'c':
                self.continuous_pars.append(self.par_features[i])
            else:
                self.discrete_pars.append(self.par_features[i])

        if self.feature_type == 'c':
            self.features = self.discrete_pars + self.continuous_pars + self.feature_name
        else:
            self.features = self.discrete_pars + self.feature_name + self.continuous_pars

        self.features = self.discrete_features + self.continuous_features
        training_df = self.training_df

        if 'd' in self.par_types:
            ''' n discrete variables will form the first n dimensions of the cpt and the (n+1)th dimension '''
            ''' it is expected that the discrete variables have been binned to numbers between 0 to n '''
            ''' if this node is discrete, it belongs to the end of discrete features list while if it is continuous'''
            ''' it will be the end of the continuous list'''

            ''' will be a kde / parametric distribution '''
            self.dims = tuple([len(np.unique) for par in self.discrete_pars] + [1])
            self.cpt = np.zeros(self.dims)

            training_df = self.training_df.groupby(self.discrete_pars)
            for group, df in training_df:
                self.cpt[group] = len(df) / float(len(self.training_df))

                try:
                    X = self.training_df[self.continuous_features]
                    kde = gaussian_kde(X.values.T, bw_method='silverman')
                except Exception:
                    '''in case of singular matrix, add gaussian noise to its principal diagonal'''
                    dim = len(X.columns)
                    mat = X.iloc[0:dim, 0:dim] + np.diag(np.random.randn(dim))
                    kde = gaussian_kde(mat.values.T, bw_method='silverman')
                self.joint_kde[group] = kde

