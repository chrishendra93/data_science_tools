import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


class ProbFunc(object):

    def __init__(self, X, features, features_type=[]):

        ''' prob func evaluates joint distribution of variables '''
        ''' three types of probfunc, mixed, discrete, continuous '''
        ''' features_type is a dictionary of feature in features and its type, i.e c or d'''
        ''' c is continuous and d is discrete'''

        self.training_df = X

        self.features = self.features_reorder(features)

        self.continuous_features = []
        self.discrete_pars = []
        self.features = features
        self.discrete_partitions = {}

        if isinstance(features_type, dict):
            self.features_types = features_type
            for feature in features_type:
                if feature not in self.par_types:
                    self.features_types[feature] = 'c'
                    self.continuous_features.append(feature)
                elif self.features_type[feature] == 'c':
                    self.continuous_features.append(feature)
                else:
                    self.discrete_features.append(feature)
        else:
            self.continuous_pars = features

        if not features_type:
            self.type = 'c'
        elif len(np.unique(features_type.values)) > 1:
            self.type = 'm'
        else:
            self.type = 'd'

        self.discrete_pars = self.var_reorder(self.discrete_pars)
        self.continuous_pars = self.var_reorder(self.continuous_pars)
        self.features = self.discrete_pars + self.continuous_pars + self.feature_name

        self.discrete_partitions = {}
        self.join_dist = {}

    def features_reorder(self, vars):
        return np.sort(vars)

    def fit_gaussian_kde(self, X):
        try:
            kde = gaussian_kde(X.values.T, bw_method='silverman')
        except Exception:
            '''in case of singular matrix, add gaussian noise to its principal diagonal'''
            dim = len(X.columns)
            mat = X.iloc[0:dim, 0:dim] + np.diag(np.random.randn(dim))
            kde = gaussian_kde(mat.values.T, bw_method='silverman')
        return kde

    def fit(self):
        if self.type == 'c':
            self.joint_dist = self.fit_gaussian_kde(self.X)
        elif self.type == 'd':
            grouped_df = self.training_df.groupby(self.features)
            for group, df in grouped_df:
                prob = len(df) / float(len(self.X))
                self.join_dist[group] = prob
        else:
            ''' the key for the joint distribution will be a tuple of the parent values'''
            ''' while the values will be a tuple of the probability and the kde of the continuous variables'''
            grouped_df = self.training_df.groupby(self.discrete_features)
            for group, df in grouped_df:
                X = df[self.continuous_features]
                kde = self.fit_gaussian_kde(X)
                prob = len(df) / float(len(self.training_df))
                self.joint_dist[group] = (prob, kde)

    def evaluate_ll(self, X, discrete_val=-1):
        if self.type == 'c':
            return self.join_dist.logpdf(X.values.T)
        else:
            if discrete_val != -1:
                ''' a fix single discrete value is assign for computation '''
                ''' expect a list of discrete values if one or more discrete values are intended '''
                ''' i.e, [1,2,3]'''
                d_val = (discrete_val,) if not isinstance(discrete_val, list) else \
                    tuple(discrete_val)

                if self.type == 'd':
                    return self.joint_dist[d_val]
                else:
                    prob = self.join_dist[d_val][0]
                    kde = self.join_dist[d_val][1]
                    return prob * kde.logpdf(X[self.continuous_features].values.T)
            else:
                if len(self.discrete_features) == 1:
                    X = X.assign(queries=X[self.discrete_features].apply(lambda x: tuple([x])))
                else:
                    X = X.assign(queries=X[self.discrete_features].apply(lambda x: tuple(x)))

                if self.type == 'd':
                    return np.array([self.joint_dist[query] for query in X.queries])
                else:
                    grouped_X = X.groupby("queries")
                    res = []
                    for group, df in grouped_X:
                        idx = df.index
                        kde = self.joint_dist[group]
                        res.append(pd.Series(kde.logpdf(df[self.continuous_features].T), index=idx))
                    return pd.concat(res).sort_index()
