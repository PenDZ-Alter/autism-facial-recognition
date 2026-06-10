import json
from ..src import vars

if __name__ == "__main__":
    print(json.dumps(vars, indent=4))