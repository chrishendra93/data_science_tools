import re
import numpy as np
from classifiers.bayesian_network.bayesian_network import BayesianNetwork


class TimeKDEBN(BayesianNetwork):

    ''' time BN is an augmented Naive Bayes model that assumes correlations between class label and time cluster'''
    ''' time cluster is a cluster of feature that is collected at the same time interval '''
    ''' in time BN, there is a directed edge from the class label to each time cluster '''
    ''' we do not assume any particular structure within the time cluster '''
    ''' but instead, we learn the joint distribution of each time cluster '''
    ''' using gaussian kernel density method via silverman's rule '''

    def __init__(self):
        super(TimeKDEBN, self).__init__()

    def fit(self, X, y, features_type={}):
        ''' we expect X to be a data frame with the following naming convention : '''
        ''' each feature column must have _i suffix where i is the time interval it is sampled at '''
        ''' i must be an integer and we expect all interval labels to be consecutive'''
        self.time_clusters = detect_time_cluster(X)
        graph = {tuple(partition_features): [self.class_label] +
                 self.time_clusters[str(int(partition) + 1)]
                 if (str(int(partition) + 1)) in self.time_clusters else [self.class_label] for
                 partition, partition_features in self.time_clusters.iteritems()}
        super(TimeKDEBN, self).fit(X, y, graph, features_type)


def detect_time_cluster(X):
    partition_labels = np.unique(map(lambda x: detect_partition_label(x), X.columns))
    time_partitions = {}
    for i in partition_labels:
        r = re.compile(".*_" + str(i))
        features = filter(r.match, X.columns)
        time_partitions[i] = features
    return time_partitions


def detect_partition_label(feature):
    idx = -1
    label = feature[-1]
    while feature[idx - 1] != "_":
        idx -= 1
        label = feature[idx] + label
    return label


if __name__ == '__main__':
    import pandas as pd
    test_df = pd.DataFrame({"a_1": np.random.randn(300), "b_1": np.random.randn(300),
                            "a_2": np.random.randn(300), "b_2": np.random.randn(300),
                            "class_label": np.append(np.ones(150), np.zeros(150))})
    X, y = test_df[["a_1", "a_2", "b_1", "b_2"]], test_df["class_label"]
    time_bn = TimeKDEBN()
    import time
    start = time.time()
    time_bn.fit(X, y)
    end = time.time()
    print "training time : " + str(end - start)
    print time_bn.time_clusters
    print time_bn.graph
    print time_bn.intermediate_results.intermediate_results
    print time_bn.compute_ll(test_df)
