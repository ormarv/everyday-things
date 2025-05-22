from utils import make_sure_dir_exists
from check_constraints import get_all_constraint_violations
from enrich_via_constraints import get_all_relations_with_labels
import csv
from relations import all_relations_lst

def print_min_max_number_of_parts(annotation_file):
    """
    Presents the minimum and maximum number of relations per object.
    Useless for the process.
    """
    total_parts = []
    min_parts_len = 10000
    max_parts_len = 0
    with open(annotation_file, "r") as tab1_info:
        tab1_data = csv.reader(tab1_info, delimiter="\t")
        for et_idx, line in enumerate(tab1_data):
            if et_idx < 1:
                continue
            parts = [p.strip() for p in line[1].split(",")]
            #print(et_idx, line)
            #print(parts)
            for part in parts:
                total_parts.append(part)
            len_parts = len(parts)
            if len_parts > max_parts_len:
                max_parts_len = len_parts
            if len_parts < min_parts_len:
                min_parts_len = len_parts

            

    print("# parts given as seed (100 everyday things):", len(total_parts))
    print("Min. # parts given as seed (100 everyday things):", min_parts_len)
    print("Max. # parts given as seed (100 everyday things):", max_parts_len)

def enrich_with_constraints_and_create_full_ET_dataset():
    make_sure_dir_exists("enriched_mms")

    et2triplets_no_enrich = {}
    ann_cnt = 0
    for transcriber_idx in range(5):
        with open("../annotation-files/transcribed-turk-ann/Diagram-transcription - id" + str(transcriber_idx) + "_tab2.tsv") as infile:
            in_file_data = csv.reader(infile, delimiter = "\t")
            for line_idx, line in enumerate(in_file_data):

                # new set 
                if line[0] != "":
                    et = line[1].lower()
                    turker = line[0]
                    et2triplets_no_enrich[(et, turker)] = []

                
                new_relation_triplet = [entry.lower().strip() for entry in line[3:6]]
                assert new_relation_triplet[1]  in all_relations_lst
                if not(new_relation_triplet[0] and new_relation_triplet[1] and new_relation_triplet[2]):
                    print(line_idx, line[3:6])
                if new_relation_triplet in et2triplets_no_enrich[(et, turker)]:
                    print(new_relation_triplet, "exists for", (et, turker), "[repeated entry]")
                    continue
                et2triplets_no_enrich[(et, turker)].append(new_relation_triplet)
                ann_cnt += 1
                
    print(ann_cnt, "annotated relations")
    print("*" * 10, "Annotated MMs", "*" * 10)
    parts = 0
    total_relations = 0
    relation_type_cnt = {"spatial": 0, "conectivity": 0, "functional": 0}
    for et, turker in et2triplets_no_enrich:
        relations = et2triplets_no_enrich[(et, turker)]
        parts_local = []
        for p1, rln, p2 in relations:

            for p in (p1, p2):
                if p not in parts_local:
                    parts_local.append(p)
            
            if rln == "connects":
                relation_type_cnt["conectivity"] += 1
            elif rln == "requires":
                relation_type_cnt["functional"] += 1
            else:
                relation_type_cnt["spatial"] += 1
            
        assert len(relations) > 1
        parts += len(parts_local)
        total_relations += len(relations)
        #print(et, turker, len(relations))
    print("Total number of MMs annotated:", len(et2triplets_no_enrich))
    print("Total # parts annotated:", parts)
    print("Total number of relations annotated:", total_relations)
    for rt in relation_type_cnt:
        print(rt, "annotated: ", relation_type_cnt[rt])

    print("Avg # parts annotated per MM:", round(parts/len(et2triplets_no_enrich), 2) )
    print("Avg. number of relations per MM:", round(total_relations/len(et2triplets_no_enrich),2))
    for rt in relation_type_cnt:
        print(rt, "annotated (avg): ", round(relation_type_cnt[rt]/len(et2triplets_no_enrich),2))

def create_full_ET_dataset(et2triplets_no_enrich, dataset_file, enrich_logfile_path, enrich_cnts_logfile_path, added_relations_logfile_path):
    mini_ET_dataset = open(dataset_file,"w")
    enrich_logfile = open(enrich_logfile_path,"w")
    enrich_cnts_logfile = open(enrich_cnts_logfile_path,"w")
    added_relations_logfile = open(added_relations_logfile_path,"w")
    mini_ET_dataset.write("\t".join(["everyday-thing", "turker", "triplet", "label"]) + "\n")
    v = 0
    t = 0
    print("Enriching with 4 types of constraints ...")
    print("*" * 10, "Annotated + Enriched MMs", "*" * 10)
    parts = []
    total_in_data = 0
    ans_true = 0
    ans_false = 0
    relation_type_cnt = {"spatial": 0, "conectivity": 0, "functional": 0}

    for et, turker in et2triplets_no_enrich:
        enrich_logfile.write(et + " " + turker + "\n")
        enrich_cnts_logfile.write(et + " " + turker + "\n")
        added_relations_logfile.write(et + " " + turker + "...")
        # enrich
        ann_relations = et2triplets_no_enrich[(et, turker)]
        relations = [] # no duplicates
        for rln in ann_relations:
            if rln not in relations:
                relations.append(rln)
        all_relations = get_all_relations_with_labels(relations, verbose = False)
        print(et, turker, "Before:", len(relations), "After:",  len(all_relations))
        
        #relations_tuples_check = {triplet[0]: triplet[1] for triplet in all_relations}
        relations_tuples_check = [triplet[0] for triplet in all_relations if triplet[1]]
        violations, total = get_all_constraint_violations(relations_tuples_check, verbose=False, max_sat_applied=False)
        v += violations
        t += total

        parts_local = []
        # store relations
        for triplet, label in all_relations:
            mini_ET_dataset.write("\t".join([et, turker, str(triplet), str(label)]) + "\n")
            total_in_data += 1
            if label:
                ans_true += 1
            else:
                ans_false += 1
                
            # get some stats
            p1, rln, p2 = triplet
            for p in (p1, p2):
                if p not in parts_local:
                    parts_local.append(p)
            
            if rln == "connects":
                relation_type_cnt["conectivity"] += 1
            elif rln == "requires":
                relation_type_cnt["functional"] += 1
            else:
                relation_type_cnt["spatial"] += 1
        parts += parts_local
            

    enrich_logfile.close() 
    enrich_cnts_logfile.close()
    added_relations_logfile.close()
    mini_ET_dataset.close()
