from classifiers.bayesian_network.bayesian_network import BayesianNetwork


class ChowLiuTree(BayesianNetwork):

    def __init__(self, X, features, graph={}, features_type={}):
        super(ChowLiuTree, self).__init__(X, features, graph, features_type)
