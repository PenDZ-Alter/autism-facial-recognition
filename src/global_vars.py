import json

def load_global_vars():
    with open('config/global_vars.jsonc', 'r') as f:
        global_vars = json.load(f)
    return global_vars

vars = load_global_vars()