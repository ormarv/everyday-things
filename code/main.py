from et_recoded import *
from recoded_imagine_a_device_with_csp import *
# In this file, no function definitions, only calls !!
# First, we create a dictionary of parts depending on et and turker.
# Structure : {(et, turker) : {“triplets”: (triplet,T/F), “parts-list”:[unique parts]}}

et2triplets_ann = load_data_and_create_et2triplets_ann()

# Then, we sort it by increasing number of parts in the mental model.
sorted_et2triplets_ann = sorted(et2triplets_ann, key=lambda et_turker: len(et2triplets_ann[et_turker]['parts-list']))

# Define a folder for all outputs.
outputs_dir = "outputs/"

# Define a threshold of confidence for propositions
filter_threshold = 50

# For each (et, turker) pair, we query macaw and select the rules using a solver
for mm_idx, et_turker in enumerate(sorted_et2triplets_ann) :
    print(et_turker, "MM #", mm_idx + 1)
        
    et, turker = et_turker
    parts_list = et2triplets_ann[et_turker]['parts-list']
    print(len(parts_list))  # functional until here
    all_result_dict = recoded_imagine_a_device_with_csp(et, turker, outputs_dir, filter_threshold, parts_list)
    print("ALL RESULTS DICT")
    print(all_result_dict)