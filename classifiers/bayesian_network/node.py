import pandas as pd
import numpy as np
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

        assert isinstance(feature_name, list) or isinstance(feature_name, tuple) or \
            isinstance(feature_name, basestring)

        assert isinstance(feature_parents, list) or isinstance(feature_parents, tuple) or \
            isinstance(feature_parents, basestring)

        self.feature_name = convert_features_to_list(feature_name)
        self.feature_parents = convert_features_to_list(feature_parents)

        self.training_df = X
        self.features_type = features_type
        self.intermediate_results = intermediate_results
        self.joint_dist = []
        self.par_dist = []
        self.marginal_dist = []

    def fit(self):
        self.joint_dist = self.intermediate_results.retrieve_joint_dist(self.feature_parents + self.feature_name)

        if len(self.feature_parents) != 0:
            self.par_dist = self.intermediate_results.retrieve_joint_dist(self.feature_parents)
            self.marginal_dist = self.intermediate_results.retrieve_joint_dist(self.feature_name)

        else:
            self.marginal_dist = self.joint_dist

    def compute_ll(self, X):
        return self.joint_dist.compute_ll(X[self.feature_name + self.feature_parents]) - \
            self.par_dist.compute_ll(X[self.feature_parents]) if len(self.feature_parents) != 0 \
            else self.marginal_dist.compute_ll(X[self.feature_name])


def convert_features_to_list(features):
    if isinstance(features, basestring):
        return [features]
    elif isinstance(features, tuple):
        return list(features)
    else:
        return features


if __name__ == '__main__':
    test = pd.DataFrame({"a": np.arange(9), "b": [0 if i < 5 else 1 for i in range(9)],
                         "c": np.random.randn(9)})
    feature_name = ["a", "b"]
    feature_parents = ["c"]
    features_type = {"a": "c", "b": "d", "c": "c"}
    prep_res = IntermediateResults(test, features_type)
    node = Node(test, feature_name, feature_parents, prep_res, features_type)
    node.fit()
    print node.joint_dist.joint_dist
    print node.compute_ll(test)
