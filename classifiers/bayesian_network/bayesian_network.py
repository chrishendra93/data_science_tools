import numpy as np
from classifiers.bayesian_network.node import Node
from classifiers.bayesian_network.intermediate_results import IntermediateResults


class BayesianNetwork(object):

    def __init__(self):
        self.training_df = []
        self.features = []
        self.graph = []
        self.features_type = []
        self.intermediate_results = []
        self.nodes = []
        self.class_label = "class_label"

    def fit(self, X, y, graph={}, features_type={}):
        self.training_df = X.assign(**{self.class_label: y})
        self.features = np.append(X.columns, self.class_label)
        self.graph = graph

        self.features_type = features_type
        self.features_type.update({self.class_label: 'd'})

        self.intermediate_results = IntermediateResults(self.training_df, self.features_type)
        self.nodes = {}

        if len(self.graph) != 0:

            for var, par in self.graph.iteritems():
                self.nodes[var] = Node(self.training_df, var, par, self.intermediate_results,
                                       self.features_type)

    def compute_entropy(self, X, n_samples=1000):
        dist = self.intermediate_results.retrieve_prob_func(X)
        samples = dist.sample(n_samples)
        return -1 * np.mean(dist.compute_ll(samples))

    def compute_mi(self, X, Y, n_samples=1000):
        ''' compute I(X; Y) based on the training_df '''
        ''' X and Y must be a list of string '''

        joint_dist = self.intermediate_results.retrieve_prob_func(X + Y)
        marginal_dist_X = self.intermediate_results.retrieve_prob_func(X)
        marginal_dist_Y = self.intermediate_results.retrieve_prob_func(Y)

        samples = joint_dist.sample(n_samples)
        return np.mean(joint_dist.compute_ll(samples) -
                       marginal_dist_X.compute_ll(samples[X]) -
                       marginal_dist_Y.compute_ll(samples[Y]))

    def compute_conditional_mi(self, X, Y, Z):
        ''' compute I(X;Y|Z)'''
        ''' I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)'''
        ''' X, Y, and Z must be a list of string'''

        return self.compute_mi(X, Y + Z) - self.compute_mi(X, Z)

    def compute_ll(self, X):
        ll = np.zeros(len(X))
        for _, node in self.nodes.iteritems():
            ll += node.compute_ll(X)
        return ll
