import csv
import ast
def load_data_and_create_et2triplets_ann():
    '''
    This function loads the original dataset and returns the et2triplets_ann dict.
        Structure : {(et, turker) : {“triplets”: (triplet,T/F), “parts-list”:[unique parts]}}
    '''
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
    return et2triplets_ann