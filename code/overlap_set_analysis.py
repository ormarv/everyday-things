import csv
import ast
import os
import pickle

def get_et2tripletslist():
    # not reusing the load_data_and_create_et2triplets_ann function because they're 
    # not exactly the same
    et2tripletslist = {}
    true_cnt = 0
    false_cnt = 0
    with open("enriched_mms/full-ET-dataset.tsv", "r") as dataset:
        lines = csv.reader(dataset, delimiter = "\t")
        for line_idx, line in enumerate(lines):
            # skip header
            if line_idx == 0:
                print("Continue")
                continue
                
            et, turker = line[0], line[1]
            if et not in et2tripletslist:
                et2tripletslist[et] = {}
                
            # per MM as in an everyday thing sketched by a turker
            et_turker = (et, turker)
            if et_turker not in et2tripletslist[et]:
                et2tripletslist[et][et_turker] = []
            
            # collect list of (triplet_tuple, True_False_label)
            triplet = ast.literal_eval(line[2])
            annotated_relation = (triplet, line[3])
            if line[3] == "True":
                true_cnt += 1
            else:
                false_cnt += 1
            assert annotated_relation not in et2tripletslist[et][et_turker]
            et2tripletslist[et][et_turker].append(annotated_relation)
        # print("true_cnt : {}".format(true_cnt))
        # print("false_cnt : {}".format(false_cnt))       
        print("Majority Class:", round(max(true_cnt, false_cnt)/(true_cnt+false_cnt) * 100,2) , "True: {}, False: {}".format(true_cnt, false_cnt))
        return et2tripletslist

def majority_class_intersection(et2tripletslist):
    true_cnt = 0
    false_cnt = 0
    et_all_intersect_relations = {}
    for everyday_thing in et2tripletslist:
        list_of_3mm = []
        for mm_by_turker in et2tripletslist[everyday_thing]:
            mm_relations = et2tripletslist[everyday_thing][mm_by_turker]
            list_of_3mm.append(mm_relations)
        assert len(list_of_3mm) == 3
        
        # converting the arrays into sets
        s1 = set(list_of_3mm[0])
        s2 = set(list_of_3mm[1])
        s3 = set(list_of_3mm[2])
        
        # calculates intersection
        set1 = s1.intersection(s2)    
        result_set = set1.intersection(s3)
            
        # convert resulting set to list
        final_list = list(result_set)
        for _, label in final_list:
            if label == "True":
                true_cnt += 1
            else:
                false_cnt += 1

        et_all_intersect_relations[everyday_thing] = final_list
        
    print("Majority Class:", round(max(true_cnt, false_cnt)/(true_cnt+false_cnt) * 100,2) , "True: {}, False: {}".format(true_cnt, false_cnt))
    return et_all_intersect_relations

def overlap_set(et_all_intersect_relations):
    et_cnt = [(entry, len(et_all_intersect_relations[entry])) for entry in et_all_intersect_relations]
    assert len(et_cnt) == 100
    cnt_et_dict = {}
    total_cnt = 0
    for et, cnt in et_cnt:
        if cnt in cnt_et_dict:
            cnt_et_dict[cnt].append(et)
        else:
            cnt_et_dict[cnt] = [et]
        total_cnt += cnt
    #cnt_et_dict
    print("# relations in overlap set:", total_cnt)

def evaluate_accuracy_for_given_belief_dict(to_evaluate, annotated_answers, verbose=False):
    correct = 0
    total = 0 
    for triplet, gold_answer in annotated_answers:
        if (gold_answer == "True" and triplet in to_evaluate) or (gold_answer == "False" and triplet not in to_evaluate):
            correct += 1
            print("MODEL CORRECT: relation", triplet, " gold answer:", gold_answer)
        else:
            if verbose:
                print("MODEL INCORRECT: relation", triplet, " gold answer:", gold_answer)

        total += 1
        
    return correct, total

def compare_accuracy_intersect(statements_dir, et_all_intersect_relations):
    before_name = "model_believe_true_props"
    after_name = "maxsat_selected_props"
    # before_name = "model_believe_true_props_filtered"
    # after_name = "maxsat_selected_props_filtered"
    improve_cnt = 0
    worsen_cnt = 0
    same_cnt = 0
    mm_total = 0
    overall_correct_cnt = {before_name:0, after_name: 0}
    overall_total_cnt = {before_name:0, after_name: 0}
    overall_improvement = 0

    acc_at_s = {before_name: {50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}, after_name: {50: 0, 60: 0, 70: 0, 80: 0, 90: 0, 100: 0}}

    # analysis
    size_of_mm_and_improvement = []

    for all_results_filename in os.listdir(statements_dir):
        if all_results_filename.startswith("."):
            continue
            
        # Annotated answers
        et, turker = all_results_filename.replace("_threshold50.pkl", "").split("_",1)
        et = et.replace("-"," ")
        annotated_answers = et_all_intersect_relations[et]
        print(et, turker)
        if len(annotated_answers) == 0:
            print(et, turker)
            continue
        
        # Model's MM
        with open(statements_dir + all_results_filename, 'rb') as f:
                all_result_dict = pickle.load(f) 
        
        # Evaluate
        correct_cnt = {before_name:0, after_name: 0}
        total_cnt = {before_name:0, after_name: 0}
        
        for prop_type in [before_name, after_name]:
            to_evaluate = all_result_dict[prop_type]
            print(prop_type)
            correct, total = evaluate_accuracy_for_given_belief_dict(to_evaluate, annotated_answers, True)
            correct_cnt[prop_type] = correct
            overall_correct_cnt[prop_type] += correct
            total_cnt[prop_type] = total
            overall_total_cnt[prop_type] += total
            
            # Calculate accuracy at s
            cur_acc = correct/total * 100
            for s in acc_at_s[before_name]:
                if cur_acc >= s:
                    acc_at_s[prop_type][s] += 1
    #         if prop_type == before_name and cur_acc < 20:
    #             print(all_results_filename)
                    
            
        assert total_cnt[before_name] == total_cnt[after_name]
        print("model_believe", "accuracy: {}/{} ({})".format(correct_cnt[before_name],\
                total_cnt[before_name],\
                round(correct_cnt[before_name]/total_cnt[before_name],2)))

        print("maxsat_selected", "accuracy: {}/{} ({})".format(correct_cnt[after_name],\
                total_cnt[after_name],\
                round(correct_cnt[after_name]/total_cnt[after_name],2)))
        
        improvement_w_maxsat = correct_cnt[after_name] - correct_cnt[before_name]
        overall_improvement += improvement_w_maxsat
        print("IMPROVEMENT W MAXSAT", "{}/{} ({})".format(improvement_w_maxsat, total_cnt[after_name],\
            round(improvement_w_maxsat/total_cnt[after_name],2)))
        size_of_mm_and_improvement.append((len(all_result_dict['macaw_predictions']), round(improvement_w_maxsat/total_cnt[after_name],2), round(correct_cnt[before_name]/total_cnt[before_name],2)))
        if improvement_w_maxsat == 0:
            same_cnt += 1
        elif improvement_w_maxsat > 0:
            improve_cnt += 1
        else:
            worsen_cnt += 1
        mm_total +=1
    print("*" * 10, "Macaw", "*" * 10)
    print(before_name, after_name)
    for before_after in acc_at_s:
        print(before_after)
        for s in acc_at_s[before_after]:
            print("acc at", s, ":", acc_at_s[before_after][s])
    print("# MMs total:", mm_total, "improved:", improve_cnt, "worsen:", worsen_cnt, "same:", same_cnt)
    print("per query overall model_believe", "accuracy: {}/{} ({})".format(overall_correct_cnt[before_name],\
            overall_total_cnt[before_name],\
            round(overall_correct_cnt[before_name]/overall_total_cnt[before_name] * 100,2)))
    print("per query overall maxsat_selected", "accuracy: {}/{} ({})".format(overall_correct_cnt[after_name],\
            overall_total_cnt[after_name],\
            round(overall_correct_cnt[after_name]/overall_total_cnt[after_name] * 100,2)))
    print("per query IMPROVEMENT W MAXSAT", "{}/{} ({})".format(overall_improvement, overall_total_cnt[after_name],\
        round(overall_improvement/overall_total_cnt[after_name] * 100,2)))
                
# et2tripletslist = get_et2tripletslist()
# et_all_intersect_relations = majority_class_intersection(et2tripletslist)
# overlap_set(et_all_intersect_relations)
# compare_accuracy_intersect("outputs/Props/",et_all_intersect_relations)