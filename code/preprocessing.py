import ast
import csv
import tqdm
from csp import *
et2triplets_ann = {}
with open("enriched_mms/full-ET-dataset.tsv", "r") as dataset:
    lines = csv.reader(dataset, delimiter = "\t")
    for line_idx, line in enumerate(lines):
        # skip header
        if line_idx == 0:
            continue
            
        # per MM as in an everyday thing sketched by a turker
        et_turker = (line[0].replace("-", " "), line[1])
        if et_turker not in et2triplets_ann:
            et2triplets_ann[et_turker] = {"triplets": [], "parts-list": []}
           
        # collect list of (triplet_tuple, True_False_label)
        triplet = ast.literal_eval(line[2])
        annotated_relation = (triplet, line[3])
        assert annotated_relation not in et2triplets_ann[et_turker]["triplets"]
        et2triplets_ann[et_turker]["triplets"].append(annotated_relation)
        
        # also collect a list of unique parts
        for part in (triplet[0], triplet[2]):
            if part not in et2triplets_ann[et_turker]["parts-list"]:
                et2triplets_ann[et_turker]["parts-list"].append(part)

# print(et2triplets_ann)

with open("macaw_query_beaker_input/input.txt", "w") as query_outfile:
    seen = {}
    for et_turker in tqdm(et2triplets_ann):
        et, turker = et_turker
        parts_list = et2triplets_ann[et_turker]["parts-list"]


        # permutation of list of parts
        perm = get_parts_perm(et, parts_list)

        # for all permutations
        for entry in perm:
            for rln in all_relations_lst:

                # form statement
                triplet = (entry[0], rln, entry[1])
                statement = triplet2statement(triplet)

                # form questions
                mc_list = ("True", "False")
                mcoptions = " ".join(["(" + chr(i+65) + ") " + word for i, word in enumerate(mc_list)])

                et = et.replace("-", " ")
                et_triplet = (et, triplet)
                if et_triplet not in seen:
                    
                    compiled_qns = "Judge whether this statement is true or false: In {determiner} {device}, {statement}.".format( \
                            determiner = get_determiner(et), device = et, statement=statement)
                    query_outfile.write("""Q: {}\nM: {}\nX: {}\nA""".format(compiled_qns, mcoptions, mcoptions))
                    query_outfile.write("\n")
                    
#                     # Ignore ngegated statements for now (Macaw may not handle negation well)
#                     neg_compiled_qns = "Judge whether this statement is true or false: In a/an {device}, it is not the case that {statement}.".format( \
#                             device = et, statement=statement)
#                     query_outfile.write(json.dumps({"id": str(et_triplet) + "-negatedTF", "question" : neg_compiled_qns, "mcoptions": mcoptions,
#                         "angle":[["question","mcoptions"],["answer"]], "explicit_outputs": mc_list}))
#                     query_outfile.write("\n")

                    seen[et_triplet] = 1
                else:
                    seen[et_triplet] += 1