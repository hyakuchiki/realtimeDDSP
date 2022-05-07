"""

For scheduling loss weights, etc.

"""
import functools

def linear_anneal(i, end_value, start_value, start, warm):
    l = max(i - start, 0)
    value = (end_value-start_value) * (float(l) / float(max(warm, l))) + start_value
    return value

class ParamSchedule():
    def __init__(self, sched_cfg):
        self.sched = {}
        for param_name, param_sched in sched_cfg.items():
            if param_name == 'name':
                continue 
            if isinstance(param_sched, float):
                self.sched[param_name] = param_sched
                continue
            if param_sched['type'] == 'linear':
                self.sched[param_name] = functools.partial(linear_anneal, 
                    start=param_sched['start'], 
                    warm=param_sched['warm'],
                    start_value=param_sched['start_v'],
                    end_value=param_sched['end_v'])
            else:
                raise ValueError()

    def get_parameters(self, cur_step):
        cur_param = {}
        for param_name, param_func in self.sched.items():
            cur_param[param_name] = param_func(i=cur_step) if callable(param_func) else param_func
        return cur_param