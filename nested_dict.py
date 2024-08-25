import numpy as np

classes_hierarchy = {
    'bg':0,
    'body':{
        'upper_body': {
            'low_hand' : 1,
            'up_hand' : 6,
            'torso' : 2,
            'head' : 4
        },
        'lower_body': {
            'low_leg': 3,
            'up_leg' : 5
        }
    }
    }

def flatten_nested_dict(input_d, output_d):
    ### return all values of subdictionaries
    res_dict = {}
    values = []
    for k,v in input_d.items():
        if not isinstance(v,dict):
            output_d[k] = [v]
            values.append(v)
        else:
            sub_values = flatten_nested_dict(v, output_d)
            output_d[k] = sub_values
            values = values + sub_values

    return values

def depth_of_keys(d):

    res = {}

    for k,v in d.items():
        res[k] = 1
        if isinstance(v,dict):
            sub_depth = depth_of_keys(v)
            for subk,subv in sub_depth.items():
                res[subk] = subv + 1

    return res

def make_classes_mapping(classes_hierarchy, exclude_keys):

    assert(isinstance(exclude_keys,list))

    values_dict = {}
    flatten_nested_dict(classes_hierarchy, values_dict)
    depth = depth_of_keys(classes_hierarchy)

    class_content = {}

    for k in values_dict:
        dep_k = depth[k]
        if f'level_{dep_k}' not in class_content:
            class_content[f'level_{dep_k}'] = {}
        if k not in exclude_keys:
            class_content[f'level_{dep_k}'][k] = values_dict[k]

    return class_content
