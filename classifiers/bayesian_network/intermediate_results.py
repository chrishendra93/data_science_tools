from classifiers.bayesian_network.joint_dist import JointDist


class IntermediateResults(object):

    def __init__(self, training_df, features_type={}):
        self.training_df = training_df
        self.intermediate_results = dict()
        self.features_type = features_type

    def _reorder_compute_dependency(func):
        def compute(self, *args, **kwargs):
            reordered_vars = sorted(*args)
            if func.__name__ + str(reordered_vars) in self.intermediate_results:
                return self.intermediate_results[func.__name__ + str(reordered_vars)]
            else:
                res = func(self, reordered_vars, **kwargs)
                self.intermediate_results[func.__name__ + str(reordered_vars)] = res
                return res

        return compute

    def _compute_dependency(func):
        def compute(self, *args, **kwargs):
            if func.__name__ + str(*args) in self.intermediate_results:
                return self.intermediate_results[func.__name__ + str(*args)]
            else:
                res = func(self, *args, **kwargs)
                self.intermediate_results[func.__name__ + str(*args)] = res
                return res

        return compute

    @_reorder_compute_dependency
    def retrieve_joint_dist(self, features):
        prob_func = JointDist(self.training_df[features], features, self, features_type=self.features_type)
        prob_func.fit()
        return prob_func
