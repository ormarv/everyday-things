#from csp import *
#from enrich_via_constraints import *
#from check_constraints import *
#from relations import *
from utils import *
import json
from tqdm import tqdm

# structure of the json file:
test_input = """{"id": "('air conditioner', ('hot coils', 'part of', 'expansion valve'))-originalTF", 
"question": "Judge whether this statement is true or false: In an air conditioner, hot coils are part of the expansion valve.", 
"mcoptions": "(A) True (B) False", 
"angle": [["question", "mcoptions"], ["answer"]], 
"explicit_outputs": ["True", "False"]}"""
test = """{"Q": "Judge whether this statement is true or false: In a tree, leaves are next to the twigs.", "\nM": "(A) True (B) False", "\nX": ["True", "False"], "\nA": ""}
"""
MODEL_NAME_OR_PATH = "allenai/macaw-large"
CUDA_DEVICES = [0,1,2]
macaw_predictions = {}
triplets_file = open("code/macaw_query_beaker_input/triples.txt")
et_triplet_2_probTF = {}
processed_results_file = open("code/macaw_query_beaker_input/processed_res.json","w")
with open("code/macaw_query_beaker_input/input.txt") as f:
    s = f.read()
    lines = s.split("A\n")
    model_dict = load_model(MODEL_NAME_OR_PATH, CUDA_DEVICES)
    i = 0
    for line in tqdm(lines):
        triple = triplets_file.readline()
        assert triple not in et_triplet_2_probTF
        et_triplet_2_probTF[triple] = {"id":triple,"answer": None, "prob_True": 0, "prob_False": 0}
        res = run_macaw(line,model_dict)
        #print(res)
        score_true = 0
        score_false = 0
        for output_choice in res['explicit_outputs']:
            if output_choice['output_text'] == "True":
                score_true = output_choice['output_prob']
            elif output_choice['output_text'] == "False":
                score_false = output_choice['output_prob']
                
        if  et_triplet_2_probTF[triple]["answer"] not in ("True", "False") and score_true > 0 and score_false > 0:
            #print(et_triplet_str, et_triplet_2_probTF[et_triplet_str]["answer"], score_true,score_false)
            et_triplet_2_probTF[triple]["answer"] = str(score_true > score_false)
            #print(et_triplet_2_probTF[et_triplet_str]["answer"])
                
        assert  et_triplet_2_probTF[triple]["answer"] == "True" or  et_triplet_2_probTF[triple]["answer"] == "False"
        
        # Scale to 100
        if score_true + score_false != 0.0:
            et_triplet_2_probTF[triple]["prob_True"] = score_true/(score_false + score_true)
            et_triplet_2_probTF[triple]["prob_False"] = 1 - et_triplet_2_probTF[triple]["prob_True"]
        else:
            # if true and false not in the top options, label_True_False_probs stays as default
            print(triple, "Alert: true and false not in the options.")
        processed_results_file.write(json.dumps(et_triplet_2_probTF[triple])+'\n')
        processed_results_file.flush()

        #i+=1
        #if i>2:
        #    break
    #json.dump(et_triplet_2_probTF,processed_results_file)