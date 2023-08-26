#prueba tcs

import unittest
from typing import Any, List, Union, Dict

ArgTypes = Union[str, List[Any], Dict[str, Any]]

data = {"exe": "bin1","args": "-o /tmp"},{"exe": "bin2","args": ["-t myTest", "-V", "-o /tmp"]},{"exe": "bin3","args":{"width":640,"height": 480}}

args = []
string = []
for i in data:
    args = i.get("args")
    if isinstance(args, str):
        string.append(i.get("exe") + " " + i.get("args"))
    else:
        if isinstance(args, list or tuple):
            con = i.get("exe")
            aux = ""
            for j in args:
                aux = aux + " " + j
            string.append(con+aux)
        else:
            con = i.get("exe")
            aux = ""
            for k in args:
                aux = aux + " " + k + " " + str(args[k])
            string.append(con+aux)
                
    
