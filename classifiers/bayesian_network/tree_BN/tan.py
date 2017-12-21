from classifiers.bayesian_network.bayesian_network import BayesianNetwork
from classifiers.bayesian_network.node import Node
from classifiers.common_miscs.graph_utils.kruskal import Kruskal
from classifiers.common_miscs.graph_utils.graph_preprocessing import convert_mst_to_edge_list
from classifiers.common_miscs.graph_utils.graph_preprocessing import convert_edge_list_to_dict
from classifiers.common_miscs.graph_utils.graph_preprocessing import create_dependencies_dict
from itertools import combinations


class TAN(BayesianNetwork):

    ''' TAN is an augmented Naive Bayes model that assumes tree-like correlation between features'''
    ''' also, just like NaiveBayes, there is a directed edge from the class label to each time feature '''

    def __init__(self):
        super(TAN, self).__init__()

    def fit(self, X, y, graph={}, features_type={}):
        super(TAN, self).fit(X, y, graph, features_type)
        if len(graph) == 0:
            vertices, edges = self.construct_mi_graph(X, y, features_type)
            mst = Kruskal({"vertices": vertices, "edges": edges}, mode="max").get_mst()
            edge_list = convert_mst_to_edge_list(mst)
            graph_dict = convert_edge_list_to_dict(edge_list, self.features)
            self.graph = create_dependencies_dict(graph_dict, self.features, self.class_label)
        for child, par in self.graph.iteritems():
            if child != self.class_label:
                self.graph[child].append(self.class_label)
        self.nodes = {var: Node(self.training_df, var, par, self.intermediate_results, self.features_type)
                      for var, par in self.graph.iteritems()}
        for _, node in self.nodes.iteritems():
            node.fit()
        return self

    def construct_mi_graph(self, X, y, features_type={}):
        vertices = {feature for feature in self.features if feature != self.class_label}
        edges = set()
        pairwise_features = list(combinations(vertices, 2))
        for v1, v2 in pairwise_features:
            edges.add((self.compute_conditional_mi([v1], [v2], [self.class_label]), v1, v2))
        return vertices, edges


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    test_df = pd.DataFrame({"a_1": np.random.randn(300), "b_1": np.random.randn(300),
                            "a_2": np.random.randn(300), "b_2": np.random.randn(300),
                            "class_label": np.append(np.ones(150), np.zeros(150))})
    X, y = test_df[["a_1", "a_2", "b_1", "b_2"]], test_df["class_label"]
    tan = TAN()
    import time
    start = time.time()
    tan.fit(X, y)
    end = time.time()
    print "training time : " + str(end - start)
    print tan.graph
    print tan.intermediate_results.intermediate_results
    print tan.compute_ll(test_df)
    pred =  tan.predict(X)
