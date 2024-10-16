class MergeTools():
    def __init__(self, merge_config):
        self.merge_method = merge_config.pop('merge_method')
        self.merge_config = merge_config
    
    def merge(self, state_dict_1, state_dict_2, state_dict_base):
        merge_function = getattr(self, self.merge_method, None)
        if callable(merge_function):
            return merge_function(state_dict_1, state_dict_2, state_dict_base)
        else:
            raise AttributeError(f"Merge method '{self.merge_method}' not found in MergeTools.")
    
    def linear(self, state_dict_1, state_dict_2, state_dict_base):
        alpha = self.merge_config.get('alpha', 0.5)
        merged_state_dict = {}
        for key in state_dict_1.keys():
            if key in state_dict_2:
                param_1 = state_dict_1[key]
                param_2 = state_dict_2[key]
                merged_param = alpha * param_1 + (1 - alpha) * param_2
                merged_state_dict[key] = merged_param
            else:
                raise KeyError(f"Parameter '{key}' not found in both state dictionaries.")
        
        return merged_state_dict