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

    def construct_mi_graphs(self):
        vertices = self.features
        edges = set()
        pairwise_features = list(combinations(self.features, 2))
        for v1, v2 in pairwise_features:
            node = Node(self.training_df, v1, v2, self.intermediate_results, self.features_type)
            edges.add((node.compute_mi(v1, v2), v1, v2))
        return vertices, edges
