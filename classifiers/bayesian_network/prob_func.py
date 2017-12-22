import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


class ProbFunc(object):

    def __init__(self, X, features, intermediate_results, features_type=[]):

        ''' prob func evaluates joint distribution of variables '''
        ''' three types of probfunc, mixed, discrete, continuous '''
        ''' features_type is a dictionary of feature in features and its type, i.e c or d'''
        ''' c is continuous and d is discrete'''

        self.training_df = X[features]
        self.intermediate_results = intermediate_results

        self.features = self.features_reorder(features)

        self.continuous_features = []
        self.discrete_features = []
        self.features = features
        self.discrete_partitions = {}
        if isinstance(features_type, dict):
            self.features_type = features_type
            for feature in features:
                if feature not in self.features_type:
                    self.continuous_features.append(feature)
                elif self.features_type[feature] == 'c':
                    self.continuous_features.append(feature)
                else:
                    self.discrete_features.append(feature)
        else:
            self.continuous_features = features

        if len(self.continuous_features) != 0 and len(self.discrete_features) != 0:
            self.type = 'm'
        elif len(self.discrete_features) != 0:
            self.type = 'd'
        else:
            self.type = 'c'

        self.discrete_features = self.features_reorder(self.discrete_features)
        self.continuous_features = self.features_reorder(self.continuous_features)
        self.features = self.discrete_features + self.continuous_features

        self.discrete_partitions = {}
        self.joint_dist = {}

    def features_reorder(self, vars):
        return sorted(vars)

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
            self.joint_dist = self.fit_gaussian_kde(self.training_df)
        elif self.type == 'd':
            grouped_df = self.intermediate_results.retrieve_groups(self.features)
            for group, df in grouped_df:
                prob = len(df) / float(len(self.training_df))
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
        X = obs[self.features]
        if self.type == 'c':
            return self.joint_dist.logpdf(X.values.T)
        else:
            if len(self.discrete_features) == 1:
                queries = X[self.discrete_features[0]]
            else:
                queries = X[self.discrete_features].apply(lambda x: tuple(x), axis=1)
            X = X.assign(queries=queries)
            if self.type == 'd':
                return np.array([self.joint_dist[query] for query in X.queries])
            else:
                grouped_X = X.groupby("queries")
                res = []
                for group, df in grouped_X:
                    idx = df.index
                    kde = self.joint_dist[group][1]
                    ll = kde.logpdf(df[self.continuous_features].values.T)
                    res.append(pd.Series(ll, index=idx))

                return pd.concat(res).sort_index()

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
            samples = np.vstack([disc_samples,cont_samples])
        sampled_df = pd.DataFrame(samples.T, columns=self.features)

        if self.type == 'm':
            sampled_df[self.discrete_features] = sampled_df[self.discrete_features].astype('int')

        return sampled_df


if __name__ == '__main__':
    from intermediate_results import IntermediateResults
    ''' demo multiple discrete and continuous vars '''
    test = pd.DataFrame({"a": np.arange(9), "b": [0 if i < 5 else 1 for i in range(9)],
                         "c": np.random.randn(9)})
    prep_res = IntermediateResults(test)
    probfunc = ProbFunc(test, ["a", "b", "c"], prep_res, {"a": "c", "b": "d"})
    print probfunc.features
    print probfunc.discrete_features
    print probfunc.continuous_features
    print probfunc.features_type
    print probfunc.type
    probfunc.fit()
    print probfunc.joint_dist
    print probfunc.compute_ll(test)
    print "---------------------------"
    test = pd.DataFrame({"a": np.arange(9), "b": [0 if i < 5 else 1 for i in range(9)],
                         "c": np.random.randn(9), "d": [0, 1, 0, 1, 0, 1, 0, 1, 0]})
    prep_res = IntermediateResults(test)
    probfunc = ProbFunc(test, ["a", "b", "c", "d"], prep_res, {"a": "c", "b": "d", "d": "d"})
    print probfunc.features
    print probfunc.discrete_features
    print probfunc.continuous_features
    print probfunc.features_type
    print probfunc.type
    probfunc.fit()
    print probfunc.joint_dist
    print probfunc.compute_ll(test)
    print probfunc.compute_ll(test)
    print probfunc.sample(5)
    print "---------------------------"
    test = pd.DataFrame({"d": [0, 1, 0, 1, 0, 1, 0, 1, 0]})
    prep_res = IntermediateResults(test)
    probfunc = ProbFunc(test, ["d"], prep_res, {"d": "d"})
    print probfunc.features
    print probfunc.discrete_features
    print probfunc.continuous_features
    print probfunc.features_type
    print probfunc.type
    probfunc.fit()
    print probfunc.joint_dist
    print probfunc.compute_ll(test)
    print probfunc.sample(2)
    print "---------------------------"
    test = pd.DataFrame({"a": np.arange(9),
                         "c": np.random.randn(9)})
    prep_res = IntermediateResults(test)
    probfunc = ProbFunc(test, ["a", "c"], prep_res, {"a": "c"})
    print probfunc.features
    print probfunc.discrete_features
    print probfunc.continuous_features
    print probfunc.features_type
    print probfunc.type
    probfunc.fit()
    print probfunc.joint_dist
    print probfunc.compute_ll(test)
    print probfunc.sample(2)
    print "---------------------------"
