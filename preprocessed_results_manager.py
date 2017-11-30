class PreprocessedResults(object):

    def __init__(self, training_df):
        self.training_df = training_df
        self.intermediate_results = dict()

    def _compute_dependency(func):
        def compute(self, *args, **kwargs):
            if func.__name__ + str(*args) in self.intermediate_results:
                return self.intermediate_results[func.__name__]
            else:
                res = func(self, *args, **kwargs)
                self.intermediate_results[func.__name__ + str(*args)] = res
                return res

        return compute

    @_compute_dependency
    def retrieve_groups(self, features):
        return self.training_df.groupby(features)
