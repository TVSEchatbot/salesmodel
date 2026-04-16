import pandas as pd

def flatten_json(data):
    flat = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            flat[name[:-1]] = x
    flatten(data)
    return pd.json_normalize(flat)