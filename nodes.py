import pandas as pd
import numpy as np
from prob_func import ProbFunc


class Nodes(object):

    ''' A node is a representation of a random variable together with its pars'''

    def __init__(self, X, features, preprocessed_results, features_type=[]):

        '''par_types are either empty, c (continuous) or d (discrete)'''
        ''' X are training data in the form of data frame'''

        if len(par_features) == 0:
            assert isinstance(X, pd.Series) or isinstance(X, pd.DataFrame), "expecting a series or a dataframe"
        else:
            assert isinstance(X, pd.Series) or isinstance(X, pd.DataFrame), "expecting a dataframe"

        self.training_df = X
        self.par_features = tuple(par_features)

        self.continuous_pars = []
        self.discrete_pars = []
        self.features = []
        self.feature_type = feature_type
        self.discrete_partitions = {}

        if isinstance(par_types, dict):
            self.par_types = par_types
            for par in par_features:
                if par not in self.par_types:
                    self.par_types[par] = 'c'
                    self.continuous_pars.append(par)
                elif self.par_types[par] == 'c':
                    self.continuous_pars.append(par)
                else:
                    self.discrete_pars.append(par)
        else:
            self.par_types = {par: 'c' for par in par_features}
            self.continuous_pars = par_features

        self.discrete_pars = self.var_reorder(self.discrete_pars)
        self.continuous_pars = self.var_reorder(self.continuous_pars)
        self.features = self.discrete_pars + self.continuous_pars + self.feature_name

        self.discrete_partitions = {}
        self.joint_cpt = []
        self.par_cpt = []
