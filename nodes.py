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

    def fit(self):
        self.par_dist = ProbFunc(self.training_df, self.feature_parents, self.intermediate_results, self.features_type)
        self.joint_dist = ProbFunc(self.training_df, self.feature_parents + [self.feature_name], self.intermediate_results,
                                   self.features_type)
        self.par_dist.fit()
        self.joint_dist.fit()

    def compute_ll(self, X, discrete_val=-1):
        return self.joint_dist.compute_ll(X, discrete_val) - self.par_dist.compute_ll(X, discrete_val)


if __name__ == '__main__':
    test = pd.DataFrame({"a": np.arange(9), "b": [0 if i < 5 else 1 for i in range(9)],
                         "c": np.random.randn(9)})
    prep_res = IntermediateResults(test)
    feature_name = "a"
    feature_parents = ["b", "c"]
    features_type = {"a": "c", "b": "d", "c": "c"}
    node = Node(test, feature_name, feature_parents, prep_res, features_type)
    node.fit()
    print node.compute_ll(test)
