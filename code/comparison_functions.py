from check_constraints import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
def convert_to_all_triplets_TF_dict(all_triplets, chosen_triplets):
    chosen_triplets_TF_dict = {}
    for triplet_TF_conf in all_triplets:
        triplet = triplet_TF_conf[0]
        if triplet in chosen_triplets:
            chosen_triplets_TF_dict[triplet] = True
        else:
            chosen_triplets_TF_dict[triplet] = False

    assert len(all_triplets) == len(chosen_triplets_TF_dict)
    return chosen_triplets_TF_dict

def compare_constraint_violations_before_after(statements_dir, all_results_filename):
    #read
    with open(statements_dir + all_results_filename, 'rb') as f:
        all_result_dict = pickle.load(f) 
    
    #model_believes_dict = convert_to_all_triplets_TF_dict(all_result_dict["macaw_predictions"], all_result_dict["model_believe_true_props"])
    model_believes_dict = all_result_dict["model_believe_true_props"]
    print("="*10, "Model's beliefs", "="*10)
    before_violate, before_total, before_tuple_n_d = get_all_constraint_violations(model_believes_dict, verbose=True, max_sat_applied=False)
    
    # maxsat_selected_dict = convert_to_all_triplets_TF_dict(all_result_dict["macaw_predictions"], all_result_dict["maxsat_selected_props"])
    maxsat_selected_dict = all_result_dict["maxsat_selected_props"]
    print("="*10, "Maxsat selected", "="*10)
    after_violate, after_total, after_tuple_n_d = get_all_constraint_violations(maxsat_selected_dict, verbose=True, max_sat_applied=True)
    print()
    
    return before_violate, before_total, before_tuple_n_d, after_violate, after_total, after_tuple_n_d
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
    
def survey(results, category_names, category_colors,SMALL_SIZE):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi = 100)
    ax.invert_yaxis()

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.75,
                label=colname, color=color)
        xcenters = starts + widths / 2


#         text_color = 'white' 
#         for y, (x, c) in enumerate(zip(xcenters, widths)):
#             ax.text(x, y, str(int(c)), ha='center', va='center',
#                     color=text_color, fontsize='small')
    ax.legend(ncol=1, bbox_to_anchor=(0., 1.04, 0.8, .102),
             loc='center', fontsize = SMALL_SIZE - 2)

    ax.axvline(x=50, color='black', linestyle='--')
    ax.set_xticks(range(0,110,10))
    ax.set_xlabel('Accuracy (%)')

    return fig, ax

