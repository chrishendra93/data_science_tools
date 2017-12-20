from classifiers.bayesian_network.bayesian_network import BayesianNetwork
from classifiers.bayesian_network.node import Node
from common_miscs.graph_utils.kruskal import Kruskal
from common_miscs.graph_utils.graph_preprocessing import convert_mst_to_edge_list, convert_edge_list_to_dict
from common_miscs.graph_utils.graph_preprocessing import create_dependencies_dict
from itertools import combinations


class ChowLiuTree(BayesianNetwork):

    def __init__(self):
        super(ChowLiuTree, self).__init__()

    def fit(self, X, y, graph={}, features_type={}):
        super(ChowLiuTree, self).fit(X, y, graph, features_type)
        if len(graph) == 0:
            vertices, edges = self.construct_mi_graphs()
            mst = Kruskal({"vertices": vertices, "edges": edges}, mode="max").get_mst()
            edge_list = convert_mst_to_edge_list(mst)
            graph_dict = convert_edge_list_to_dict(edge_list, self.features)
            self.graph = create_dependencies_dict(graph_dict, self.features, self.class_label)
        self.nodes = {var: Node(self.training_df, var, par, self.intermediate_results, self.features_type)
                      for var, par in self.graph.iteritems()}
        for _, node in self.nodes.iteritems():
            node.fit()

    def construct_mi_graphs(self):
        vertices = self.features
        edges = set()
        pairwise_features = list(combinations(self.features, 2))
        for v1, v2 in pairwise_features:
            # node = Node(self.training_df, v1, [v2], self.intermediate_results, self.features_type)
            # node.fit()
            edges.add((self.compute_mi([v1], [v2]), v1, v2))
        return vertices, edges


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    test_df = pd.DataFrame({"a": np.random.randn(200), "b": np.random.randn(200),
                            "c": np.random.randn(200), "d": np.random.randn(200),
                            "e": np.random.randn(200), "f": np.random.randn(200),
                           "class_label": np.append(np.ones(100), np.zeros(100))})
    X, y = test_df[["a", "b", "c", "d", "e", "f"]], test_df["class_label"]
    c_tree = ChowLiuTree()
    import time
    start = time.time()
    c_tree.fit(X, y, features_type={"a": "c", "b": "c"})
    end = time.time()
    print c_tree.features
    print "training time : " + str(end - start)
    print c_tree.graph
    print c_tree.nodes
    print c_tree.nodes['class_label'].joint_dist.joint_dist
    print c_tree.intermediate_results.intermediate_results
    start = time.time()
    print c_tree.compute_mi(["a"], ["b"])
    end = time.time()
    print "duration for computing mi : " + str(end - start)
    print c_tree.compute_conditional_mi(["a", "e", "class_label"], ["b"], ["c"])
    start = time.time()
    print c_tree.compute_ll(test_df)
    end = time.time()
    print "duration for computing ll : " + str(end - start)
