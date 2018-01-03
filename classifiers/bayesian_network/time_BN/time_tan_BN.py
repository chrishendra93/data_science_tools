import numpy as np
from itertools import product
from classifiers.bayesian_network.bayesian_network import BayesianNetwork
from classifiers.bayesian_network.tree_BN.chow_liu_tree import ChowLiuTree
from classifiers.bayesian_network.time_BN.time_kde_BN import detect_time_cluster
from classifiers.bayesian_network.time_BN.time_kde_BN import detect_partition_label
from common_miscs.graph_utils.dijkstra import dijkstra
from common_miscs.graph_utils.graph_preprocessing import merge_dependencies_dict


class TimeTANBN(BayesianNetwork):

    ''' time BN is an augmented Naive Bayes model that assumes correlations between class label and time cluster'''
    ''' time cluster is a cluster of feature that is collected at the same time interval '''
    ''' in time BN, there is a directed edge from the class label to each time cluster '''
    ''' we do not assume any particular structure within the time cluster '''
    ''' but instead, we learn the joint distribution of each time cluster '''
    ''' using gaussian kernel density method via silverman's rule '''

    def __init__(self):
        super(TimeTANBN, self).__init__()

    def fit(self, X, y, graph={}, features_type={}):
        ''' we expect X to be a data frame with the following naming convention : '''
        ''' each feature column must have _i suffix where i is the time interval it is sampled at '''
        ''' i must be an integer and we expect all interval labels to be consecutive from 0 to len(partition)'''
        super(TimeTANBN, self).init_vars(X, y, graph, features_type)
        self.time_clusters = detect_time_cluster(X)
        self.n_partitions = len(self.time_clusters)

        parents_set = self.find_parents_set()
        roots_set = self.construct_roots_set(parents_set)
        self.chow_liu_clusters = self.construct_chow_liu_clusters(X, features_type, roots_set)
        graph = {}

        for partition, tree in self.chow_liu_clusters.iteritems():
            graph = merge_dependencies_dict(graph, tree.graph)

        self.graph = merge_dependencies_dict(graph, parents_set)

        for _, node in roots_set.iteritems():
            self.graph[node].append(self.class_label)

        self.graph[self.class_label] = []
        super(TimeTANBN, self).fit(X, y, graph=self.graph, features_type=features_type,
                                   intermediate_results=self.intermediate_results)

        return self

    def construct_chow_liu_clusters(self, X, features_type, roots_set):
        chow_liu_clusters = {}
        for partition, partition_features in self.time_clusters.iteritems():
            root = roots_set[partition]
            chow_liu_tree = ChowLiuTree()
            chow_liu_clusters[partition] = chow_liu_tree.fit(X[partition_features], y=[],
                                                             features_type=features_type, root=root,
                                                             intermediate_results=self.intermediate_results)
        return chow_liu_clusters

    def construct_parents_DAG(self):
        edge_list_dags = []
        for i in range(self.n_partitions):
            if i != self.n_partitions - 1:
                edges_pair = product(self.time_clusters[str(i)], self.time_clusters[str(i + 1)])
                for v1, v2 in edges_pair:
                    edge_weight = -self.compute_mi([v1, self.class_label], [v2]) + \
                        self.compute_entropy([v2])
                    edge_list_dags.append((edge_weight, v1, v2))
        return edge_list_dags

    def find_parents_set(self):
        edge_list_dags = self.construct_parents_DAG()

        first_partition = self.time_clusters['0']
        last_partition = self.time_clusters[str(self.n_partitions - 1)]
        source_dst_pairs = product(first_partition, last_partition)
        shortest_paths = [dijkstra(edge_list_dags, v1, v2) for v1, v2 in source_dst_pairs]
        parents_pair = shortest_paths[0][1]

        return {prev_par: [next_par] for prev_par, next_par in parents_pair}

    def construct_roots_set(self, parents_set):
        nodes = set(parents_set.keys() + [item for x in parents_set.values() for item in x])
        return {detect_partition_label(node): node for node in nodes}


if __name__ == '__main__':
    import pandas as pd
    test_df = pd.DataFrame({"a_0": np.random.randn(3000), "b_0": np.random.randn(3000),
                            "a_1": np.random.randn(3000), "b_1": np.random.randn(3000),
                            "a_2": np.random.randn(3000), "b_2": np.random.randn(3000),
                            "class_label": np.append(np.ones(1500), np.zeros(1500))})
    X, y = test_df[["a_0", "a_1", "a_2", "b_0", "b_1", "b_2"]], test_df["class_label"]
    time_bn = TimeTANBN()
    import time
    start = time.time()
    time_bn.fit(X, y)
    end = time.time()
    print "training time : " + str(end - start)
    print time_bn.graph
    print time_bn.compute_ll(test_df)
    pred = time_bn.predict(X)
    print pred
