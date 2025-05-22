#from get_constraints import *
from utils import make_sure_dir_exists
import pickle
import os
from recoded_run_query_macaw_everyday_thing_batch_mode import recoded_run_query_macaw_everyday_thing_batch_mode
from csp import run_maxsat, filter_props
# no_order_object_parts_propositions seems weird
# find out the use of device and turker for store_wcnf (maybe we don't store it, for now at least?)
# samples = [{"id": "('car', ('roof', 'contains', 'wheels'))\n", "answer": "False", "prob_True": 0.24557940825536556, "prob_False": 0.7544205917446345}]
def recoded_imagine_a_device_with_csp(device, turker, outputs_dir, filter_threshold, parts=[]):
    device = device.lower()
    tag = "threshold" + str(filter_threshold)
    
    lm_query_dir = outputs_dir + "LMResponses/" # dir where you want to save macaw output
    wcnf_dir = outputs_dir + "WCNF_format/" # dir where you want to save these wcnf for reference
    plots_dir = outputs_dir + "VizPlots/" # dir where you want to store output files
    statements_dir = outputs_dir + "Props/"# dir where you save data from this run
    all_results_filename = device.replace(" ", "-") + "_" + turker + "_" + tag
    for desired_dir in [outputs_dir, lm_query_dir, wcnf_dir, plots_dir, statements_dir]:
        make_sure_dir_exists(desired_dir)
    if all_results_filename + ".pkl" in os.listdir(statements_dir):
         # read
        with open(statements_dir + all_results_filename + ".pkl", 'rb') as f:
             all_result_dict = pickle.load(f)
        print("Read from file ...", len(all_result_dict["macaw_predictions"]), "triplets ...")
    else:
        # lm response - do not want to query LM again if the same device has been asked
        if device.replace(" ", "-") + "-" + turker + ".pkl" in os.listdir(lm_query_dir):
            # read
            with open(lm_query_dir + device.replace(" ", "-") + "-" + turker + ".pkl", 'rb') as f:
                 triplet_ans_conf_lst, triplet_ans_conf_lst_true, neg_ans_conf_lst = pickle.load(f) 
        else:
            # query macaw
            triplet_ans_conf_lst, triplet_ans_conf_lst_true, neg_ans_conf_lst = recoded_run_query_macaw_everyday_thing_batch_mode(device, parts)
            # save
            with open(lm_query_dir + device.replace(" ", "-") + "-" + turker + ".pkl", 'wb') as f:
                pickle.dump([triplet_ans_conf_lst, triplet_ans_conf_lst_true, neg_ans_conf_lst], f)
        # use maxsat
        print("Running maxsat ...", len(triplet_ans_conf_lst), "triplets...")
        model_believe_true_props, maxsat_selected_props = run_maxsat(device, turker, wcnf_dir, triplet_ans_conf_lst, neg_ans_conf_lst, triplet_ans_conf_lst_true, use_only_model_true_props = False)

        print("Filtering ...", len(model_believe_true_props), "triplets...", len(maxsat_selected_props), "triplets...")
        # filter based on confidence
        model_believe_true_props_filtered = filter_props(model_believe_true_props, filter_threshold)
        maxsat_selected_props_filtered = filter_props(maxsat_selected_props, filter_threshold)
        all_result_dict = {"macaw_predictions": triplet_ans_conf_lst,\
                            "macaw_predictions_believe_true": triplet_ans_conf_lst_true,\
                            "model_believe_true_props": model_believe_true_props,\
                            "maxsat_selected_props": maxsat_selected_props,\
                            "filter_threshold": filter_threshold,\
                            "model_believe_true_props_filtered": model_believe_true_props_filtered,\
                            "maxsat_selected_props_filtered": maxsat_selected_props_filtered}
        with open(statements_dir + all_results_filename + ".pkl", 'wb') as f:
            pickle.dump(all_result_dict, f)
    return all_result_dict
