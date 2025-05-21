from comparison_functions import *
import os
#from et_recoded import load_data_and_create_et2triplets_ann
# all the functions in this file work!!!

def compare_violations_before_after(statements_dir):
    before_violate_cnt =0
    before_total_cnt = 0

    after_violate_cnt = 0
    after_total_cnt = 0

    n_d_by_type = {"before": {"type1": [0,0], "type2": [0,0], "type3": [0,0], "type4": [0,0]},\
                "after": {"type1": [0,0], "type2": [0,0], "type3": [0,0], "type4": [0,0]}}

    before_mm_w_violate_cnt = 0
    after_mm_w_violate_cnt = 0

    for all_results_filename in os.listdir(statements_dir):
        if all_results_filename.startswith("."):
            continue
            
        print(all_results_filename)
        print("*" * 15, all_results_filename.rsplit("_threshold",1)[0], "*" * 15)
        before_violate, before_total, before_tuple_n_d, after_violate, after_total, after_tuple_n_d = compare_constraint_violations_before_after(statements_dir, all_results_filename)
        
        before_violate_cnt += before_violate
        before_total_cnt += before_total
        if before_violate > 0:
            before_mm_w_violate_cnt += 1
        for type_idx, (n, d) in enumerate(before_tuple_n_d):
            n_d_by_type["before"]["type" + str(type_idx + 1)][0] += n 
            n_d_by_type["before"]["type" + str(type_idx + 1)][1] += d 
            
        after_violate_cnt += after_violate
        after_total_cnt += after_total
        if after_violate > 0:
            after_mm_w_violate_cnt += 1
        for type_idx, (n, d) in enumerate(after_tuple_n_d):
            n_d_by_type["after"]["type" + str(type_idx + 1)][0] += n 
            n_d_by_type["after"]["type" + str(type_idx + 1)][1] += d 
    print("*" * 10,"Macaw-11B", "*" * 10)
    print("Number of MMs", len(os.listdir(statements_dir)))
    print("TOTAL VIOLATE COUNT before maxsat {}/{} ({})".format(before_violate_cnt, before_total_cnt, round(before_violate_cnt/before_total_cnt * 100, 2)))

    print("TOTAL VIOLATE COUNT after maxsat {}/{} ({})".format(after_violate_cnt, after_total_cnt, round(after_violate_cnt/after_total_cnt * 100, 2)))

    # by constraint type
    for before_after in n_d_by_type:
        print(before_after, "maxsat:")
        per_type_avg_n = 0
        per_type_avg_d = 0
        for contraint_type in n_d_by_type[before_after]:
            cur_n, cur_d = n_d_by_type[before_after][contraint_type]
            if cur_d == 0:
                print(contraint_type, "violate count : {}/{}".format(cur_n, cur_d))
                print("Not applicable for given mental model!")
            else:
                print(contraint_type, "violate count : {} ({}/{})".format(round(cur_n/cur_d * 100, 2), cur_n, cur_d))
                per_type_avg_n += cur_n/cur_d * 100
                per_type_avg_d += 1
        print("macro avg (equal weight to each category):", round(per_type_avg_n/per_type_avg_d, 2))


    print("MMs with VIOLATE before maxsat {}/{} ({})".format(before_mm_w_violate_cnt, len(os.listdir(statements_dir)), round(before_mm_w_violate_cnt/len(os.listdir(statements_dir)) * 100, 2)))

    print("MMs with VIOLATE after maxsat {}/{} ({})".format(after_mm_w_violate_cnt, len(os.listdir(statements_dir)), round(after_mm_w_violate_cnt/len(os.listdir(statements_dir)) * 100, 2)))


def evaluate_accuracy_for_given_belief_dict(to_evaluate, annotated_answers, verbose=False, relation_analysis=False):
    correct = 0
    total = 0 
    rln2_correct_total = {}
    for triplet, gold_answer in annotated_answers:
        
        rln = triplet[1]
        if rln not in rln2_correct_total:
            rln2_correct_total[rln] = {"correct": 0, "total": 0}
            
        if (gold_answer == "True" and triplet in to_evaluate) or (gold_answer == "False" and triplet not in to_evaluate):
            correct += 1
            print("MODEL CORRECT: relation", triplet, " gold answer:", gold_answer)
            rln2_correct_total[rln]["correct"] += 1
        else:
            if verbose:
                print("MODEL INCORRECT: relation", triplet, " gold answer:", gold_answer)

        total += 1
        rln2_correct_total[rln]["total"] += 1
        
    if relation_analysis:
        return correct, total, rln2_correct_total
    else:
        return correct, total
    
def compare_accuracy_before_after(statements_dir,et2triplets_ann):
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
    et2_correct_total_overall = {before_name: {}, after_name: {}}
    rln2_correct_total_overall = {before_name: {}, after_name: {}}

    # analysis
    size_of_mm_and_improvement = []

    for all_results_filename in os.listdir(statements_dir):
        if all_results_filename.startswith("."):
            continue
            
        # Annotated answers
        et, turker = all_results_filename.replace("_threshold50.pkl", "").split('_',1)
        print(et, turker)
        annotated_answers = et2triplets_ann[(et.replace("-"," "), turker)]['triplets']
        print(et, turker)
        
        # Model's MM
        with open(statements_dir + all_results_filename, 'rb') as f:
                all_result_dict = pickle.load(f) 
        
        # Evaluate
        correct_cnt = {before_name:0, after_name: 0}
        total_cnt = {before_name:0, after_name: 0}
        
        for prop_type in [before_name, after_name]:
            to_evaluate = all_result_dict[prop_type]
            print(prop_type)
            correct, total, rln2_correct_total = evaluate_accuracy_for_given_belief_dict(to_evaluate, annotated_answers, verbose=True, relation_analysis=True)
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

            # Analyze by relationship
            for rln in rln2_correct_total:
                if rln not in rln2_correct_total_overall[prop_type]:
                    rln2_correct_total_overall[prop_type][rln] = rln2_correct_total[rln]
                else:
                    rln2_correct_total_overall[prop_type][rln]["correct"] += rln2_correct_total[rln]["correct"]
                    rln2_correct_total_overall[prop_type][rln]["total"] += rln2_correct_total[rln]["total"]
            
            #  Analyze by ET
            if et not in et2_correct_total_overall[prop_type]:
                et2_correct_total_overall[prop_type][et] = {"correct": 0, "total": 0}
            et2_correct_total_overall[prop_type][et]["correct"] += correct
            et2_correct_total_overall[prop_type][et]["total"] += total
                    
            
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
        '''if "gpt3" in model_name:
            # (num_props, improvement, before_maxsat)
            size_of_mm_and_improvement.append((len(all_result_dict['gpt3_predictions']), round(improvement_w_maxsat/total_cnt[after_name],2), round(correct_cnt[before_name]/total_cnt[before_name],2)))
        else:'''
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
    return acc_at_s, mm_total, rln2_correct_total_overall, et2_correct_total_overall
    

def plot_correct_mms_for_acc(rcParams,SMALL_SIZE,MEDIUM_SIZE,acc_at_s,mm_total):
    plt.rcParams['figure.figsize'] = rcParams
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)

    # naming the x and y axis
    plt.xlabel('Accuracy @ s')
    plt.ylabel('Correct mental models (%)')
    for before_after in acc_at_s:
        # print(acc_at_s[before_after])
        acc_at_s_tuples = sorted(acc_at_s[before_after].items())
        x, y = zip(*acc_at_s_tuples) 
        # print(x,[y_cnt/mm_total * 100 for y_cnt in y])
        plt.plot(x, [y_cnt/mm_total * 100 for y_cnt in y], "o-", label=before_after)
        
    diff_acc_at_s_y = []
    diff_acc_at_s_x = []
    for acc_s in acc_at_s['model_believe_true_props']:
        diff_acc_at_s_y.append((acc_at_s['maxsat_selected_props'][acc_s] - acc_at_s['model_believe_true_props'][acc_s])/mm_total * 100)
        diff_acc_at_s_x.append(acc_s)
    plt.plot(diff_acc_at_s_x, diff_acc_at_s_y, "ko--", label="improvement")

    plt.legend(['Base LM', 'Base LM + constraint reasoning', 'Improvement w. constraint reasoning'], fontsize=18)

    #plt.axhline(y = 50, color = 'k', linestyle = 'dashed')
    plt.yticks(range(0, 110, 20))
    plt.tight_layout()
    # plt.savefig(plots_dir + "/" + model_name + "_" + "acc_at_s.png")
    plt.show()

def plot_improvement_acc(rcParams,SMALL_SIZE,MEDIUM_SIZE, rln2_correct_total_overall):
    # Doesn't work in notebook
    # TODO : debug
    plt.rcParams['figure.figsize'] = rcParams
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)

    #“Base LM”, “Base LM + constraint reasoning”
    category_names = ['Base LM', 'Improvement w. constraint reasoning']
    category_colors = ['cornflowerblue', 'peachpuff']
    results = {}

    for relation in all_relations_lst:
        accuracy_before = rln2_correct_total_overall['model_believe_true_props'][relation]["correct"]/rln2_correct_total_overall['model_believe_true_props'][relation]["total"]
        percent_correct_before = round(accuracy_before * 100, 2)
        
        accuracy_after = rln2_correct_total_overall['maxsat_selected_props'][relation]["correct"]/rln2_correct_total_overall['maxsat_selected_props'][relation]["total"]
        improvement = accuracy_after - accuracy_before
        percent_improvement = round(improvement * 100, 2)

        results[relation] = [percent_correct_before, percent_improvement]
    survey(results, category_names, category_colors, SMALL_SIZE)
    plt.tight_layout()
    # plt.savefig(plots_dir + "/" + model_name + "_" + "acc_by_relation.png")
    plt.show()

def results_by_improvement(et2_correct_total_overall, category_names:list, category_colors:list):
    '''
    Parameters:
    category
    '''
    results = {}
    et_100 = [et for et in sorted(list(et2_correct_total_overall['model_believe_true_props'].keys()))]
    for et in et_100:
        accuracy_before = et2_correct_total_overall['model_believe_true_props'][et]["correct"]/et2_correct_total_overall['model_believe_true_props'][et]["total"]
        percent_correct_before = round(accuracy_before * 100, 2)
        
        accuracy_after = et2_correct_total_overall['maxsat_selected_props'][et]["correct"]/et2_correct_total_overall['maxsat_selected_props'][et]["total"]
        improvement = accuracy_after - accuracy_before
        percent_improvement = round(improvement * 100, 2)

        results[et.replace("-", " ")] = [percent_correct_before, percent_improvement]
    results_by_improvement = {k: v for k, v in sorted(results.items(), key=lambda item: item[1][0], reverse=True)}
    results_by_improvement_lists = {1 : {}, 2: {}, 3: {}, 4: {}, 5: {}}

    for idx_et, et in enumerate(results_by_improvement):
        if results_by_improvement[et][1] < 0: 
            results_by_improvement_lists[idx_et//20 + 1][et] = [results_by_improvement[et][0] + results_by_improvement[et][1], 0, -results_by_improvement[et][1]]
        else:
            results_by_improvement_lists[idx_et//20 + 1][et] = results_by_improvement[et] + [0]
    return results_by_improvement_lists

def plot_by_improvement_and_category(cat:int, results_by_improvement_lists,plots_dir, SMALL_SIZE):
    assert(cat<=20 and cat>0)
    category_names = ['Base LM', 'Improvement w. constraint reasoning', "Worsen w. constraint reasoning"]
    category_colors = ['cornflowerblue', 'peachpuff', 'gray']
    survey(results_by_improvement_lists[cat], category_names, category_colors, SMALL_SIZE)
    plt.tight_layout()
    plt.savefig(plots_dir + "Macaw_acc_by_et_best_initial{}.png".format(cat*20))
    plt.show()
    
def plot_by_improvement_merged(rcParams,results_by_improvement_lists,plots_dir, SMALL_SIZE):
    merged = {**results_by_improvement_lists[2], **results_by_improvement_lists[3], **results_by_improvement_lists[4]}
    plt.rcParams['figure.figsize'] = rcParams
    category_names = ['Base LM', 'Improvement w. constraint reasoning', "Worsen w. constraint reasoning"]
    category_colors = ['cornflowerblue', 'peachpuff', 'gray']
    survey(merged, category_names, category_colors, SMALL_SIZE)
    plt.tight_layout()
    plt.savefig(plots_dir + "Macaw_acc_by_et_accuracy-middle21-80.png")
    plt.show()

'''statements_dir = "outputs/Props/"
plots_dir = "plots/"
compare_violations_before_after(statements_dir)
et2triplets_ann = load_data_and_create_et2triplets_ann()
acc_at_s, mm_total, rln2_correct_total_overall, et2_correct_total_overall = compare_accuracy_before_after(statements_dir,et2triplets_ann)

rcParams = 12, 10
SMALL_SIZE = 25
MEDIUM_SIZE = 30
category_names = ['Model believe', 'Improvement w. MaxSAT']
category_colors = ['cornflowerblue', 'peachpuff']
plot_correct_mms_for_acc(rcParams,SMALL_SIZE, MEDIUM_SIZE, acc_at_s, mm_total)
plot_improvement_acc(rcParams, SMALL_SIZE, MEDIUM_SIZE, rln2_correct_total_overall)
results_by_improvement_lists = results_by_improvement(et2_correct_total_overall,category_names, category_colors)
plot_by_improvement_and_category(1, results_by_improvement_lists, plots_dir, SMALL_SIZE)
plot_by_improvement_merged(rcParams, results_by_improvement_lists, plots_dir, SMALL_SIZE)'''