import itertools
from relations import *
import spacy
from utils import *
MODEL_NAME_OR_PATH = "allenai/macaw-large"
CUDA_DEVICES = [0,1,2]
macaw_getMM_logfile = open("saved_logs/pipeline_step1_macaw_getMM_log.txt", "w")
impose_contraints_logfile = open("saved_logs/pipeline_step2_impose_contraints_log.txt", "w")

nlp = spacy.load("en_core_web_sm")

def is_plural(noun):
    """
    Input: the entity (string)
    Output: whether the entity is plural (True/False)
    """
    doc = nlp(noun)
    lemma = " ".join([token.lemma_ for token in doc])
    plural = True if (noun is not lemma and noun.endswith("s")) else False
    return plural

def triplet2statement(triplet):
    relation_singular_plural_dict = {"has part": "have part", \
                                    "contains": "contain", \
                                    "surrounds": "surround", \
                                    "requires": "require"
                                    }

    num = "plural" if is_plural(triplet[0]) else "singular"

    if triplet[1] == "connects":
        linking_phrase = "directly connected to"
    else:
        linking_phrase = triplet[1] # default is singular

    if linking_phrase in relation_singular_plural_dict:
        # special treatment of linking phrase for plural
        if num == "plural":
            linking_phrase = relation_singular_plural_dict[linking_phrase]
    else:
        # + is/are
        add = "are" if num == "plural" else "is"
        linking_phrase = add + " " + linking_phrase

    if "has part" == linking_phrase or "have part" == linking_phrase:
        phrase_p1, phrase_p2 = linking_phrase.split(" ")
        its_their = "their" if is_plural(triplet[0]) else "its"
        statement = " ".join([triplet[0], phrase_p1, "the", triplet[2], "as", its_their, phrase_p2])
    else:
        statement = " ".join([triplet[0], linking_phrase, "the", triplet[2]])
    return statement

def get_parts_perm(device, parts=[]):
    """
    Input: everyday thing, (optional) list of parts
    Output: permutation of list of parts
    """
    macaw_getMM_logfile.write("Everything processed: " + device + "\n")
    '''if not parts:
        qns_parts = "What are the parts of a/an {}?".format(device)
        ans_parts = ask_macaw(qns_parts)'''
        #macaw_getMM_logfile.write("\t".join(["[===QA===] ", str(qns_parts), str(ans_parts)]) + "\n")
        #parts = list(set([part.strip() for part in ans_parts.strip().split(",")]))
    macaw_getMM_logfile.write("List of parts: \n" + str(parts) + "\n")
    perm = list(itertools.permutations(parts, 2))
    macaw_getMM_logfile.write("Permutation of parts: \n" + str(perm) + "\n")
    
    return perm

def ask_macaw_conf(question, mc_list = ["yes", "no"], mc_str ="", get_all=False):
    '''
    input: question in the form of string
    output: answer
    default MC options is Yes/No
    Use this function to get answer from macaw with probabilities (slower queries)
    '''
    model_dict = load_model(MODEL_NAME_OR_PATH, CUDA_DEVICES)
    if mc_list:
        mcoptions = " ".join(["(" + chr(i+65) + ") " + word for i, word in enumerate(mc_list)])
        input = "Q: " + question + "\nM: " + mcoptions + "\nX: " + mcoptions + "\nA"
    elif mc_str:
        input = "Q: " + question + "\nM: " + mc_str + "\nX: " + mc_str + "\nA"
    response = run_macaw(input,model_dict)

    if get_all:
         ans_score =  [(x['output_text'], x['output_prob']) for x in response['explicit_outputs']]
    else:
        ans_score = [(x['output_text'], x['output_prob']) for x in response()['explicit_outputs'] if x['output_text'] == response.json()['output_slots_list'][0]['answer']][0]
    
    
    return ans_score

def get_p_statement_and_p_neg_statement(device, statement):
    '''
    Output: direct query answer, p_statement, p_neg_statement
    '''
    probs_dict = {"T_given_statement": 0, "T_given_neg_statement": 0}

    mc_list = ("True", "False")
    #mcoptions = " ".join(["(" + chr(i+65) + ") " + option for i, option in enumerate(mc_list)])
    
    compiled_qns = "Judge whether this statement is true or false: In a/an {device}, {statement}.".format( \
                    device = device, statement=statement)
    ans_conf_all = ask_macaw_conf(compiled_qns, mc_list, get_all=True)
    ans = ans_conf_all[0][0] if ans_conf_all[0][1] > ans_conf_all[1][1] else ans_conf_all[1][0]
    macaw_getMM_logfile.write("\t".join(["[===QA===] ", compiled_qns, ans, str(ans_conf_all)]) + "\n")
    for entry in ans_conf_all:
        if entry[0] == "True":
            probs_dict["T_given_statement"] = entry[1]

    neg_compiled_qns = "Judge whether this statement is true or false: In a/an {device}, it is not the case that {statement}.".format( \
                    device = device, statement=statement)
    neg_ans_conf_all = ask_macaw_conf(neg_compiled_qns, mc_list, get_all=True) # T_given_neg_statement, F_given_neg_statement
    macaw_getMM_logfile.write("\t".join(["[===QA===] ", neg_compiled_qns, str(neg_ans_conf_all)]) + "\n")
    for entry in neg_ans_conf_all:
        if entry[0] == "True":
            probs_dict["T_given_neg_statement"] = entry[1]

    return ans, probs_dict["T_given_statement"], 1 - probs_dict["T_given_statement"]

def get_statements_that_macaw_believesT(triplet_ans_conf_lst):
    triplet_ans_conf_lst_true = []
    for triplet, ans, ans_conf in triplet_ans_conf_lst:
        if ans == "True":
            triplet_ans_conf_lst_true.append([triplet,  ans, ans_conf])
    return triplet_ans_conf_lst_true

def recoded_run_query_macaw_everyday_thing_batch_mode(device,parts):
    # TODO reuse the other files I wrote but take out the device as a key this time
    # example of a good line :  
    # [('yolk', 'has part', 'egg white'), 'False', 0.34120868304130925]
    # (God, this is a mess)
    # begin by getting a list of all the permutations of the parts list
    perm = get_parts_perm(device, parts)

    # then initialize the lists that contain the triplets and the negative answer confidences
    triplet_ans_conf_lst = [] # list of list
    neg_ans_conf_lst = [] # list
    for entry in perm:
        for rln in all_relations_lst:
            
            triplet = (entry[0], rln, entry[1])
            et_triplet_str = str((device, triplet))
            statement = triplet2statement(triplet)
            ans, p_statement, p_neg_statement = get_p_statement_and_p_neg_statement(device, statement)
            triplet_ans_conf_lst.append([triplet,  ans, p_statement])
            neg_ans_conf_lst.append(p_neg_statement)
    triplet_ans_conf_lst_true = get_statements_that_macaw_believesT(triplet_ans_conf_lst)
    return triplet_ans_conf_lst, triplet_ans_conf_lst_true, neg_ans_conf_lst