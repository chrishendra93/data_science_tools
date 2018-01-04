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
        self.classes = []
        self.nodes = []
        self.class_label = "class_label"
        self.init = False

    def fit(self, X, y=[], graph={}, features_type={}, intermediate_results=None):

        if not self.init:
            self.init_vars(X, y, graph, features_type, intermediate_results)

        if len(self.graph) != 0:
            for var, par in self.graph.iteritems():
                self.nodes[var] = Node(self.training_df, var, par, self.intermediate_results,
                                       self.features_type)
                self.nodes[var].fit()

    def init_vars(self, X, y, graph, features_type, intermediate_results=None):
        self.features_type = features_type

        if len(y) != 0:
            self.training_df = X.assign(**{self.class_label: y})
            self.features = np.append(X.columns, self.class_label)
            self.classes = np.unique(y)
            self.features_type.update({self.class_label: 'd'}) if len(y) != 0 else None
        else:
            self.training_df = X
            self.features = X.columns

        self.graph = graph

        if not intermediate_results:
            self.intermediate_results = IntermediateResults(self.training_df, self.features_type)
        else:
            self.intermediate_results = intermediate_results

        self.nodes = {}
        self.init = True

    def compute_entropy(self, X, n_samples=1000):
        dist = self.intermediate_results.retrieve_joint_dist(X)
        samples = dist.sample(n_samples)
        return -1 * np.mean(dist.compute_ll(samples))

    def compute_mi(self, X, Y, n_samples=1000):
        ''' compute I(X; Y) based on the training_df '''
        ''' X and Y must be a list of string '''

        joint_dist = self.intermediate_results.retrieve_joint_dist(X + Y)
        marginal_dist_X = self.intermediate_results.retrieve_joint_dist(X)
        marginal_dist_Y = self.intermediate_results.retrieve_joint_dist(Y)
        sorted_vars = np.sort(X + Y).tolist()
        X_idx = [sorted_vars.index(x) for x in X]
        Y_idx = [sorted_vars.index(y) for y in Y]
        samples = joint_dist.sample(n_samples)
        return np.mean(joint_dist.compute_ll(samples) -
                       marginal_dist_X.compute_ll(samples[:, X_idx]) -
                       marginal_dist_Y.compute_ll(samples[:, Y_idx]))

    def compute_conditional_mi(self, X, Y, Z):
        ''' compute I(X;Y|Z)'''
        ''' I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)'''
        ''' X, Y, and Z must be a list of string'''

        return self.compute_mi(X, Y + Z) - self.compute_mi(X, Z)

    def compute_ll(self, obs):
        ll = np.zeros(len(obs))
        for _, node in self.nodes.iteritems():
            ll += node.compute_ll(obs)
        return ll

    def predict(self, X):
        return np.argmax(np.array([self.compute_ll_class(X, label) for label in self.classes]).T, axis=1)

    def compute_ll_class(self, X, label):
        return self.compute_ll(X.assign(**{self.class_label: label * np.ones(len(X))}))

    def show_factorisation(self):
        factorisation = "p(class) "
        for var, par in self.graph.iteritems():
            if par:
                factorisation += "p({} | {}, class) ".format(var, par)
        return factorisation

    def score(self):
        if len(self.classes) == 0:
            raise ValueError("not applicable")
        else:
            y_true, y_pred = self.training_df[self.class_label], self.predict(self.training_df[self.features])
            return np.sum(y_true == y_pred) / float(len(y_pred))
