import pandas as pd
import numpy as np
from prob_func import ProbFunc
from intermediate_results import IntermediateResults


class Node(object):

    ''' A node is a representation of a random variable together with its pars'''

    def __init__(self, X, feature_name, feature_parents, intermediate_results, features_type={}):

        ''' A node object holds joint distribution of ProbFunc class '''
        ''' To compute P(X|par_X) = P(X,par_X) / P(par_x)'''
        ''' Node object therefore has to store two joint distribution '''
        ''' X are training data in the form of data frame'''

        if len(feature_parents) == 0:
            assert isinstance(X, pd.Series) or isinstance(X, pd.DataFrame), "expecting a series or a dataframe"
        else:
            assert isinstance(X, pd.DataFrame), "expecting a dataframe"

        self.training_df = X
        self.features_type = features_type
        self.feature_name = feature_name
        self.feature_parents = feature_parents
        self.intermediate_results = intermediate_results
        self.joint_dist = []
        self.par_dist = []
        self.marginal_dist = []

    def fit(self):
        self.joint_dist = ProbFunc(self.training_df, self.feature_parents + [self.feature_name], self.intermediate_results,
                                   self.features_type)
        if len(self.feature_parents) != 0:
            self.par_dist = ProbFunc(self.training_df, self.feature_parents, self.intermediate_results,
                                     self.features_type)
            self.marginal_dist = ProbFunc(self.training_df, [self.feature_name], self.intermediate_results,
                                          self.features_type)

            self.par_dist.fit()
            self.marginal_dist.fit()
        else:
            self.marginal_dist = self.join_dist
        self.joint_dist.fit()

    def compute_mi(self, n_samples=1000):
        '''compute I(feature; parents)'''
        if len(self.feature_parents) != 0:
            samples = self.joint_dist.sample(n_samples)
            return np.mean(self.joint_dist.compute_ll(samples) - self.marginal_dist.compute_ll(samples) -
                           self.par_dist.compute_ll(samples))
        else:
            raise ValueError("root node has no MI to compute")

    def compute_conditional_mi(self):
        return

    def compute_ll(self, X):
        return self.joint_dist.compute_ll(X) - self.par_dist.compute_ll(X)


if __name__ == '__main__':
    test = pd.DataFrame({"a": np.arange(9), "b": [0 if i < 5 else 1 for i in range(9)],
                         "c": np.random.randn(9)})
    prep_res = IntermediateResults(test)
    feature_name = "a"
    feature_parents = ["b", "c"]
    features_type = {"a": "c", "b": "d", "c": "c"}
    node = Node(test, feature_name, feature_parents, prep_res, features_type)
    node.fit()
    print node.joint_dist.joint_dist
    print node.compute_ll(test)
    print node.compute_mi()
