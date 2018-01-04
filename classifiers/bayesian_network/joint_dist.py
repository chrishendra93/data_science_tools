import numpy as np
import numpy_indexed as npi
import pandas as pd
from scipy.stats import gaussian_kde


class JointDist(object):

    def __init__(self, X, features, intermediate_results, features_type=[]):

        ''' prob func evaluates joint distribution of variables '''
        ''' three types of probfunc, mixed, discrete, continuous '''
        ''' features_type is a dictionary of feature in features and its type, i.e c or d'''
        ''' c is continuous and d is discrete'''
        ''' joint dist will store every computation as numpy array'''
        ''' and convert features name into indices '''
        assert len(self.features) != 0

        self.intermediate_results = intermediate_results
        self.features = features

        self.features_idx = self.assign_features_idx()
        self.training_data = self.reformat_data_input(X)

        self.features_type = features_type
        self.continuous_features, self.discrete_features = self.assign_features_type()
        self.type = self.assign_dist_type()

        self.continuous_ordering = self.assign_features_ordering(self.continuous_features)
        self.discrete_ordering = self.assign_features_ordering(self.discrete_features)

        self.features_ordering = self.continuous_ordering + self.discrete_ordering
        self.training_data = self.reorder_cols(self.reformat_data_input(X), self.features_ordering)

        self.discrete_partitions = {}
        self.joint_dist = {}

    def assign_features_idx(self):
        return {self.features[i]: i for i in range(len(self.features))}

    def reformat_data_input(self, X):
        if isinstance(X, pd.DataFrame):
            assert len(X.columns) == len(self.features)
            return X[self.features].values
        elif isinstance(X, np.ndarray):
            ''' if the input type is numpy array, we assume that it matches the features '''
            ''' the numpy array must be of shape (n_dim, n_samples) '''
            assert X.shape[1] == len(self.features)
            return X
        elif isinstance(X, pd.Series):
            return X.values
        else:
            raise ValueError("invalid input type")

    def assign_features_type(self):
        continuous_features, discrete_features = [], []
        if isinstance(self.features_type, dict):
            for feature in self.features:
                if feature not in self.features_type:
                    continuous_features.append(feature)
                elif self.features_type[feature] == 'c':
                    continuous_features.append(feature)
                else:
                    discrete_features.append(feature)
        else:
            continuous_features = self.features

        return continuous_features, discrete_features

    def assign_dist_type(self):
        if len(self.continuous_features) != 0 and len(self.discrete_features) != 0:
            return 'm'
        elif len(self.discrete_features) != 0:
            return 'd'
        else:
            return 'c'

    def assign_features_ordering(self, features):
        return [self.features_idx[feature] for feature in features]

    def reorder_cols(self, arr, features_order):
        assert isinstance(arr, np.ndarray)
        assert isinstance(features_order, np.ndarray) or isinstance(features_order, list)

        if len(features_order) == 1:
            return arr[:, features_order[0]]
        else:
            return arr[:, features_order]

    def fit_gaussian_kde(self, X):
        kde = []
        dim = len(X.columns)
        mat = X
        while True:
            try:
                kde = gaussian_kde(mat.values.T, bw_method='silverman')
            except Exception:
                '''in case of singular matrix, add gaussian noise to its principal diagonal'''
                mat = X.iloc[0:dim, 0:dim] + np.diag(np.random.randn(dim))
                continue
            break
        return kde

    def fit(self):
        if self.type == 'c':
            self.joint_dist = self.fit_gaussian_kde(self.training_data)
        elif self.type == 'd':
            grouped_df = self.intermediate_results.retrieve_groups(self.features)
            for group, df in grouped_df:
                prob = len(df) / float(len(self.training_data))
                self.joint_dist[group] = prob
        else:
            ''' the key for the joint distribution will be a tuple of the parent values'''
            ''' while the values will be a tuple of the probability and the kde of the continuous variables'''
            grouped_df = self.intermediate_results.retrieve_groups(self.discrete_features)
            for group, df in grouped_df:
                X = df[self.continuous_features]
                kde = self.fit_gaussian_kde(X)
                prob = len(df) / float(len(self.training_df))
                self.joint_dist[group] = (prob, kde)

    def compute_ll(self, obs):
        X = self.reorder_cols(self.reformat_data_input(obs), self.features_ordering)
        if self.type == 'c':
            return self.joint_dist.logpdf(X.T)
        else:
            if self.type == 'd':
                if len(self.discrete_features) == 1:
                    return np.array([self.joint_dist[query] for query in X])
                else:
                    return np.array([self.joint_dist[tuple(query)] for query in X])
            else:
                grouped_X = npi.group_by(X[:, self.discrete_ordering])
                keys = grouped_X.unique
                idx = grouped_X.inverse
                vals = grouped_X.split(X[:, self.continuous_ordering])
                res = np.array([])
                for i in range(len(keys)):
                    group = keys[i] if len(self.discrete_features) == 1 else tuple(keys[i])
                    prob = self.joint_dist[group][0]
                    kde = self.joint_dist[group][1]
                    ll = prob + kde.logpdf(vals[i].T)
                    res = np.append(res, ll)

                return res[idx]

    def sample(self, n_samples=1000):
        samples = []
        if self.type == 'c':
            samples = self.joint_dist.resample(n_samples)
        elif self.type == 'd':
            groups = np.array(self.joint_dist.keys())
            probs = np.array(self.joint_dist.values())
            idx = np.arange(len(probs))
            rand_indx = np.random.choice(idx, size=n_samples, p=probs)
            samples = groups[rand_indx]
        else:
            '''perform ancestral sampling technique'''
            '''we sample discrete values then we sample from the conditional distribution of the'''
            '''continuous functions. i.e, if X is continuous and Y is discrete, then we sample Y '''
            '''from P(Y) then sample X from f(X|Y)'''
            groups = np.array(self.joint_dist.keys())
            probs = np.array([x[0] for x in self.joint_dist.values()])
            idx = np.arange(len(probs))
            rand_idx = np.random.choice(idx, size=n_samples, p=probs)
            unique_groups, n_samples_arr = np.unique(rand_idx, return_counts=True)
            cont_samples = np.hstack([self.joint_dist[tuple(groups[unique_groups[i]])][1].resample(n_samples_arr[i])
                                     if len(self.discrete_features) > 1 else
                                     self.joint_dist[groups[unique_groups[i]]][1].resample(n_samples_arr[i])
                                     for i in range(len(unique_groups))])
            disc_samples = groups[np.repeat(unique_groups, n_samples_arr)].T
            samples = np.vstack([disc_samples, cont_samples]).T

        if self.type == 'm':
            sampled_df[self.discrete_features] = sampled_df[self.discrete_features].astype('int')

        return sampled_df
