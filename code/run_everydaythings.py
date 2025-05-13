#from csp import *
#from enrich_via_constraints import *
#from check_constraints import *
#from relations import *
from utils import *

# structure of the json file:
test_input = """{"id": "('air conditioner', ('hot coils', 'part of', 'expansion valve'))-originalTF", 
"question": "Judge whether this statement is true or false: In an air conditioner, hot coils are part of the expansion valve.", 
"mcoptions": "(A) True (B) False", 
"angle": [["question", "mcoptions"], ["answer"]], 
"explicit_outputs": ["True", "False"]}"""
test = """{"Q": "Judge whether this statement is true or false: In a tree, leaves are next to the twigs.", "\nM": "(A) True (B) False", "\nX": ["True", "False"], "\nA": ""}
"""

with open("code/macaw_query_beaker_input/input.txt") as f:
    s = f.read()
    lines = s.split("A\n")
    test = lines[0]
    print(test)

MODEL_NAME_OR_PATH = "allenai/macaw-large"
CUDA_DEVICES = [0,1,2]
input = test
model_dict = load_model(MODEL_NAME_OR_PATH, CUDA_DEVICES)
res = run_macaw(input,model_dict)
for r in res['explicit_outputs']:
    answer = r['output_text']
    prob = r['output_prob']
    print("{} ({})".format(answer,prob))