# -*- coding: utf-8 -*-
"""QA_F1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14u7tksn4quqR_cUlTF1uq-qIzxKGvRBt

# 1. Setup
"""

from google.colab import drive
drive.mount('/content/drive')

import xlrd, os, json, statistics
import pandas as pd
from pandas import ExcelWriter
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# !pip install --upgrade xlrd

os.chdir("drive/MyDrive/NLP Project")
print(os.getcwd())
from Colabs import confusion_matrix

def get_all_dfs(xlsx_path):
  """ Convert from an xlsx file to a dictionary of dataframes.
  
  Returns: 
    dict: model labels as keys and dataframes as values

  """
  xls = pd.ExcelFile(xlsx_path)
  sheet_names = xls.sheet_names
  all_dfs = {}
  for sheet in sheet_names:
    all_dfs[sheet] = pd.read_excel(xlsx_path, sheet_name = sheet)
  return all_dfs

import copy
def hs_correct(all_HS_dfs):

  pr_corrected = "data/HS-program-corrected.xls"
  df_pr = pd.read_excel(pr_corrected, sheet_name = "HS program corrected")[["idx", "extracted term", "start", "end", "corrected", "full text"]]
  sv_corrected = "data/HS-service-corrected.xls"
  df_sv = pd.read_excel(sv_corrected, sheet_name = "HS service corrected")[["idx", "extracted term", "start", "end", "corrected", "full text"]]

  corrected_dfs = {"Program": [],
                   "Service": [],
                   "Need Satisfier": [],
                   "Need": []}
  all_HS_dfs_copy = copy.deepcopy(all_HS_dfs)

  for i, row in all_HS_dfs["Program"].iterrows():
    idx, start, end = row.at["idx"], row.at["start"], row.at["end"]
    cor_row = df_pr[(df_pr["idx"] == idx) & (df_pr["start"] == start) & (df_pr["end"] == end)].iloc[0]
    corrected = cor_row.at["corrected"]

    if corrected in ["Service", "Need Satisfier", "Need"]:
      corrected_dfs[corrected].append(row)

    if (corrected != "flag") & (type(corrected) != float):
      all_HS_dfs_copy["Program"].drop(i)


  for i, row in all_HS_dfs["Service"].iterrows():
    idx, start, end = row.at["idx"], row.at["start"], row.at["end"]
    cor_row = df_sv[(df_sv["idx"] == idx) & (df_sv["start"] == start) & (df_sv["end"] == end)].iloc[0]
    corrected = cor_row.at["corrected"]

    if corrected == "Program":
      corrected_dfs[corrected].append(row)
      all_HS_dfs_copy["Service"].drop(i)

  for entity in corrected_dfs:
    add_df = pd.DataFrame(corrected_dfs[entity], columns=["idx", "extracted term", "start", "end", "full text"])
    all_HS_dfs_copy[entity] = pd.concat([all_HS_dfs_copy[entity], add_df])
  
  return all_HS_dfs_copy

mapping = {"program name": ["Program", "Organization"], 
           "client": ["Client", "Eligibility", "BeneficialStakeholder", "Community"],
           "need satisfier": ["Need Satisfier", "Service"],
           "need outcome": ["Outcome", "Need"],
           "catchment area": ["CatchmentArea"]}

# qa_layer0 = get_all_dfs("data/QA_results/QA layer/answers/sample test/HS_layer0_eval.xlsx")
# qa_layer1 = get_all_dfs("data/QA_results/QA layer/answers/HS June/evals/HS_layer1_eval.xlsx")
# qa_layer1_new = get_all_dfs("data/QA_results/QA layer/answers/HS_new_layer1_eval.xlsx")
# qa_layer0_agg = pd.concat(list(qa_layer0.values()))
# qa_layer1_agg = pd.concat(list(qa_layer1.values()))
# qa_layer1_new_agg = pd.concat(list(qa_layer1_new.values()))

# qa_layer0_original = get_all_dfs("data/QA_results/QA layer/answers/sample test/HS_layer0_original_eval.xlsx")
# qa_layer1_original = get_all_dfs("data/QA_results/QA layer/answers/sample test/HS_layer1_original_eval.xlsx")
# qa_layer0_fixed = get_all_dfs("data/QA_results/QA layer/answers/sample test/HS_layer0_fixed_eval.xlsx")
# qa_layer1_fixed = get_all_dfs("data/QA_results/QA layer/answers/sample test/HS_layer1_fixed_eval.xlsx")
# qa_layer0_original_agg = pd.concat(list(qa_layer0_original.values()))
# qa_layer1_original_agg = pd.concat(list(qa_layer1_original.values()))
# qa_layer0_fixed_agg = pd.concat(list(qa_layer0_fixed.values()))
# qa_layer1_fixed_agg = pd.concat(list(qa_layer0_fixed.values()))

qa_layer0_fixed = get_all_dfs("data/QA_results/QA layer/answers/rank_results/HS_layer0_fixed_eval.xlsx")
qa_layer1_fixed = get_all_dfs("data/QA_results/QA layer/answers/rank_results/HS_layer1_fixed_eval.xlsx")




# ground_truth = get_all_dfs("data/HS_June5_all.xlsx")
# new_ground_truth = {}
# for entity in mapping:
#   new_list = []
#   for cat in mapping[entity]:
#     new_list.append(ground_truth[cat])
#   new_ground_truth[entity] = pd.concat(new_list)

ground_truth_corrected = get_all_dfs("data/HS_June5_all_corrected.xlsx")
new_ground_truth_corrected = {}
for entity in mapping:
  new_list2 = []
  for cat in mapping[entity]:
    new_list2.append(ground_truth_corrected[cat])
  new_ground_truth_corrected[entity] = pd.concat(new_list2)

new_ground_truth.keys()

qa_layer1_new["program name"]

"""## Old functions & Testing"""

qa_layer0["need satisfier"][qa_layer0["need satisfier"]['eval'] == "Y1"]

list(ground_truth["Availability"]["extracted term"])
print(ground_truth.keys())
print(qa_layer0.keys())
print(qa_layer1.keys())

# False negative count
def fn_count(truth, correct_truth):

  FN = len(set(truth)) - len(set(truth) & set(correct_truth))

  return FN

def tp_fp_count(truth, correct_data, correct_truth, wrong_data, ver):

  if ver == 0:
    TP = len(correct_truth)
    FP = len(wrong_data)

  if ver == 1:
    TP = len(set(correct_truth)) 
    FP = len(wrong_data) 
  
  if ver == 2:
    TP = len(set(correct_truth)) 
    FP = len(set(wrong_data)) 
  
  if ver == 3:
    TP = len(set(correct_data)) 
    FP = len(set(wrong_data)) 


  return TP, FP

truth = ["a", "b", "c", "d"] #GT
correct_data = ["A", "C", "A", "aa", "aa"] #Yx
correct_truth = ["a", "c", "a", "a", "a"] #"truth"
wrong_data = ["z", "k", "j", "u", "u", "j"] #N
assert fn_count(truth, correct_truth) == 2
assert tp_fp_count(truth, correct_data, correct_truth, wrong_data, ver=0) == (5, 6)  # adds up to all answers
assert tp_fp_count(truth, correct_data, correct_truth, wrong_data, ver=1) == (2, 6)  # ...
assert tp_fp_count(truth, correct_data, correct_truth, wrong_data, ver=2) == (2, 4)  # adds up to all unique response, but for correct response (TP) we are counting unique truth values
assert tp_fp_count(truth, correct_data, correct_truth, wrong_data, ver=3) == (3, 4)  # adds up to all unique response

def choose_eval(method):
  e = None
  if method == "Exact":
    e = ["Y1"]
  elif method == "Partial1":
    e = ["Y1", "Y2"]
  elif method == "Partial2":
    e = ["Y1", "Y2", "Y3"]
  return e

def confusion_matrix(ground_truth, eval_results, method, entities, qs = None):
  cm = {"TP": 0, "FP": 0, "FN": 0}
  e = choose_eval(method)
  for entity in entities:
    idxs = eval_results[entity]["idx"].unique()

    for idx in idxs:
      gt_df = new_ground_truth[entity]
      result_df = eval_results[entity]

      true_values = list(gt_df[gt_df["idx"] == int(idx)]["extracted term"])
      this_result = result_df[result_df["idx"] == idx]

      if qs is not None:
        this_result = this_result[this_result["qid"] == qs]

      correct_extracted = []
      correct_truth = []
      wrong_extracted = []

      for i, row in this_result.iterrows():
        if row.at["eval"] in e:
          correct_extracted.append(row.at["response"])
          correct_truth.append(row.at["truth"])
        else:
          wrong_extracted.append(row.at["response"])
      
      TP = len(correct_truth)
      FP = len(wrong_extracted)
      FN = len(set(true_values)) - len(set(true_values) & set(correct_truth))
      cm["TP"] += TP
      cm["FP"] += FP
      cm["FN"] += FN

  return cm

"""# Helper functions & F1 calculation"""

def calculate_f1(cm):
  try:
    precision = cm["TP"] / (cm["TP"] + cm["FP"])
  except ZeroDivisionError:
    precision = 0
  
  try:
    recall = cm["TP"] / (cm["TP"] + cm["FN"])
  except ZeroDivisionError:
    recall = 0
  
  try:
    f1 = (2 * precision * recall) / (precision + recall)
  except ZeroDivisionError:
    f1 = 0

  return f1

def get_all_idxs(ground_truth, qs):
  all_idxs = []
  for entity in ground_truth.keys():
    all_idxs += list(ground_truth[entity]["idx"])
  return list(set(all_idxs))


def filter_by_qs(eval_results, qs):
  new_eval_results = {}
  for entity in eval_results:
    curr_df = eval_results[entity]
    new_df = curr_df[curr_df["qid"] == qs]
    new_eval_results[entity] = new_df
  return new_eval_results


def filter_by_idx(dfs, idx):
  new_dfs = {}
  for entity in dfs:
    curr_df = dfs[entity]
    new_df = curr_df[curr_df["idx"] == idx]
    new_dfs[entity] = new_df
  return new_dfs

"""# Layer 0,1 by questions"""

gt_mapping = {"pr": "program name", "cl": "client", "ns": "need satisfier", "no": "need outcome", "ca": "catchment area"}

def cm_layer_qs(ground_truth, eval_results, match_type, qid):
  new_eval_results = filter_by_qs(eval_results, qid) #dic of df
  if len(qid) == 6:
    qcode = qid[3:5]
  else:
    qcode = qid.replace("/","").split("_")[-2]

  new_ground_truth = ground_truth[gt_mapping[qcode]] #df
  # print(new_ground_truth)
  all_idxs = get_all_idxs(ground_truth, qid)
  all_cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

  for idx in all_idxs:
    this_results = filter_by_idx(new_eval_results, idx)
    this_gt = new_ground_truth[new_ground_truth["idx"] == idx] #df
    this_gt_dict = {gt_mapping[qcode]: this_gt}
    all_ground_truth = {}
    for ent in ground_truth:
      filtered = ground_truth[ent][ground_truth[ent]["idx"] == idx]
      all_ground_truth[ent] = filtered

    this_cm = confusion_matrix.get_confusion_matrix(this_gt_dict, this_results, match_type, all_ground_truth=all_ground_truth)

    
    for entity in this_cm:
      all_cm["TP"] += this_cm[entity][0]
      all_cm["FP"] += this_cm[entity][1]
      all_cm["FN"] += this_cm[entity][2]
      all_cm["TN"] += this_cm[entity][3]

  return all_cm

"""## Layer 0"""

import tqdm

layer = qa_layer0_fixed
matches = ["exact", "partial1", "partial2"]
qs = {"pr_": 7, "cl_": 11, "ns_": 4, "n/o_": 6, "ca_": 4}
match = matches[1]


for match in matches:
  res = []
  # for q in tqdm.tqdm(qs):
  for q in qs:
    for i in range(1, qs[q]+1):
      qid = q + str(i)
      cm = cm_layer_qs(new_ground_truth_corrected, layer, match, qid)
      cm['precision'] = cm["TP"] / (cm["TP"] + cm["FP"])
      cm['recall'] = cm["TP"] / (cm["TP"] + cm["FN"])
      cm['f1'] = calculate_f1(cm)
      cm['qid'] = qid
      cm['Entity'] = gt_mapping[qid.replace("/","").split("_")[-2]]
      cm['Match'] = match
      res.append(cm.copy())
  pd.DataFrame(res).to_csv(f"data/QA_results/QA layer/answers/rank_results/layer0_{match}_fixed_qids.csv", index=False)

"""## Layer 1"""

import tqdm
layer = qa_layer1_fixed
matches = ["exact", "partial1", "partial2"]
givs = {"pr_": [0, 2, 2, 2, 2], "cl_": [2, 0, 2, 2, 1], "ns_": [2, 2, 0, 2, 1], "no_": [2, 2, 2, 0, 1], "ca_": [1, 2, 2, 2, 0]}
tars = ["pr", "cl", "ns", "no", "ca"]

for match in tqdm.tqdm(matches):
  res = []
  for giv in givs:
    for i in range(5):
      for j in range(givs[giv][i]):
        qid = giv + tars[i] + str(j)
        cm = cm_layer_qs(new_ground_truth_corrected, layer, match, qid)
        cm['precision'] = cm["TP"] / (cm["TP"] + cm["FP"])
        cm['recall'] = cm["TP"] / (cm["TP"] + cm["FN"])
        cm['f1'] = calculate_f1(cm)
        cm['qid'] = qid
        cm['Entity'] = gt_mapping[tars[i]]
        cm['Match'] = match
        res.append(cm.copy())

  pd.DataFrame(res).to_csv(f"data/QA_results/QA layer/answers/rank_results/layer1_{match}_fixed_qids.csv", index=False)

import pandas as pd
import numpy as np
matches = ["exact", "partial1", "partial2"]
for match in matches:
  f = f"data/final results/QA/metrics/layer1_{match}_fixed_qids.csv"
  df = pd.read_csv(f)
  acc = (df["TP"] + df["TN"])/(df["TP"] + df["TN"] + df["FP"] + df["FN"])
  mcc = ((df["TP"] * df["TN"] - df["FP"] * df["FN"])/
               np.sqrt((df["TP"] + df["FP"]) * 
                       (df["TP"] + df["FN"]) * 
                       (df["TN"] + df["FP"]) * 
                       (df["TN"] + df["FN"])))
  df = df.assign(accuracy=acc)
  df = df.assign(mcc=mcc)
  df.to_csv(f"data/final results/QA/metrics/layer1_{match}_fixed_qids.csv", index=False)

"""# Layer 0 Aggregate"""

exact_rank = {"program name": [1, 6, 2, 5, 4, 3, 7],
              "client": [4, 3, 11, 10, 1, 2, 7, 8, 6, 5, 9],
              "need satisfier": [1, 4, 3, 2],
              "need outcome": [3, 1, 5, 6, 2, 4],
              "catchment area": [2, 4, 1, 3]}
partial1_rank = {"program name": [6, 1, 5, 4, 2, 3, 7],
                 "client": [4, 11, 10, 2, 7, 8, 6, 3, 1, 9, 5],
                 "need satisfier": [4, 1, 2, 3],
                 "need outcome": [5, 1, 3, 6, 2, 4],
                 "catchment area": [4, 1, 3, 2]}
partial2_rank = {"program name": [1, 6, 5, 2, 4, 7, 3],
                 "client": [4, 11, 10, 2, 7, 8, 6, 3, 1, 9, 5],
                 "need satisfier": [4, 1, 2, 3],
                 "need outcome": [5, 3, 1, 6, 2, 4],
                 "catchment area": [4, 3, 2, 1]}
qid_mapping_layer0 = {"program name": "pr_",
               "client": "cl_",
               "need satisfier": "ns_",
               "need outcome": "n/o_",
               "catchment area": "ca_"}

def choose_rank_layer0(method):
  rank = None
  if method == "exact":
    rank = exact_rank
  elif method == "partial1":
    rank = partial1_rank
  elif method == "Partial2":
    rank = partial2_rank
  return rank

def aggregate_TP_FP(df_eval_qid, qid, rank, e):

  qid_rank = [qid + str(i) for i in rank]
  df_eval_qid["qid"] = df_eval_qid["qid"].astype("category")
  df_eval_qid["qid"] = df_eval_qid["qid"].cat.set_categories(qid_rank)
  sorted_eval = list(df_eval_qid.sort_values(["qid"])["eval"])
  
  try:
    rep_eval = sorted_eval[0]
  except IndexError:
    rep_eval = "N"
  if rep_eval in e:
    TP, FP = 1, 0
  else:
    TP, FP = 0, 1
  return TP, FP

def layer0_aggregate(ground_truth, eval_results, method, entities):
  cm = {"TP": 0, "FP": 0, "FN": 0}
  e = choose_eval(method)
  rank = choose_rank_layer0(method)

  for entity in entities:
    idxs = eval_results[entity]["idx"].unique()
    this_rank = rank[entity]

    for idx in idxs:
      gt_df = new_ground_truth[entity]
      result_df = eval_results[entity]

      true_values = list(gt_df[gt_df["idx"] == int(idx)]["extracted term"])
      this_result = result_df[result_df["idx"] == idx]
      correct_truth = list(this_result[this_result["eval"].isin(e)]["truth"])

      df_eval_qid = this_result[["eval", "qid"]]
      qid = qid_mapping_layer0[entity]


      TP, FP = aggregate_TP_FP(df_eval_qid, qid, rank, e)
      FN = len(set(true_values)) - len(set(true_values) & set(correct_truth))
      cm["TP"] += TP
      cm["FP"] += FP
      cm["FN"] += FN

  return cm

layer1_exact_rank = ["cl_pr0", "ns_pr1", "no_pr1", "ca_pr0", 
                     "pr_cl0", "ns_cl0", "no_cl0", "ca_cl1", 
                     "pr_ns0", "cl_ns1", "no_ns0", "ca_ns0", 
                     "pr_no0", "cl_no1", "ns_no1", "ca_no0", 
                     "pr_ca1", "cl_ca0", "ns_ca0", "no_ca0"]

layer1_part1_rank = ["cl_pr0", "ns_pr1", "no_pr1", "ca_pr0", 
                     "pr_cl0", "ns_cl0", "no_cl1", "ca_cl0", 
                     "pr_ns1", "cl_ns1", "no_ns1", "ca_ns1", 
                     "pr_no1", "cl_no1", "ns_no0", "ca_no1", 
                     "pr_ca0", "cl_ca0", "ns_ca0", "no_ca0"]

layer1_part2_rank = ["cl_pr0", "ns_pr1", "no_pr1", "ca_pr0", 
                     "pr_cl0", "ns_cl0", "no_cl1", "ca_cl0", 
                     "pr_ns1", "cl_ns1", "no_ns1", "ca_ns1", 
                     "pr_no1", "cl_no1", "ns_no1", "ca_no1", 
                     "pr_ca1", "cl_ca0", "ns_ca0", "no_ca0"]

mapping_layer1 = {"program name": "pr",
               "client": "cl",
               "need satisfier": "ns",
               "need outcome": "no",
               "catchment area": "ca"}

def choose_rank_layer1(method):
  rank = None
  if method == "Exact":
    rank = layer1_exact_rank
  elif method == "Partial1":
    rank = layer1_part1_rank
  elif method == "Partial2":
    rank = layer1_part2_rank
  return rank


def aggregate_TP_FP_layer1(df_eval_qid, qid, rank, e):

  rep_qid = None
  rep_eval = "N"
  for r in rank:
    if qid in r:
      rep_qid = r
      rep_eval = list(df_eval_qid[df_eval_qid["qid"] == rep_qid]["eval"])
      print(rep_eval)

  if rep_eval in e:
    TP, FP = 1, 0
  else:
    TP, FP = 0, 1
  return TP, FP

a = qa_layer1_agg[qa_layer1_agg["idx"] == 114.0]
b = a[a["qid"] == "cl_pr0"]
b

def layer1_aggregate(ground_truth, eval_results, method, giv_entities, res_entities, dim):
  cm = {"TP": 0, "FP": 0, "FN": 0}
  e = choose_eval(method)
  rank = choose_rank_layer1(method)

  if dim == "given":
    for giv_entity in giv_entities:
      giv_id = mapping_layer1[giv_entity]
      result_df = eval_results[eval_results["qid"].str.contains(giv_id+"_")]
      idxs = result_df["idx"].unique()

      for res_entity in res_entities:
        res_id = mapping_layer1[res_entity]
        gt_df = new_ground_truth[res_entity]
        result_df = result_df[result_df["qid"].str.contains("_"+res_id)]

        for idx in idxs:
          true_values = list(gt_df[gt_df["idx"] == int(idx)]["extracted term"])
          this_result = result_df[result_df["idx"] == idx]
          correct_truth = list(this_result[this_result["eval"].isin(e)]["truth"])

          qid = giv_id + "_" + res_id

          TP, FP = aggregate_TP_FP_layer1(this_result, qid, rank, e)
          FN = len(set(true_values)) - len(set(true_values) & set(correct_truth))
          cm["TP"] += TP
          cm["FP"] += FP
          cm["FN"] += FN

  return cm

# qa_layer1_agg[qa_layer1_agg["qid"][:2] == "cl"]
qa_layer1_agg[qa_layer1_agg["qid"].str.contains("cl"+"_")]

all_ent = ['program name', 'client', 'need satisfier', 'need outcome', 'catchment area']
match = "Exact"
dim = "given"

cm = layer1_aggregate(ground_truth, qa_layer1_agg, match, all_ent[:], all_ent[:], dim)
print(calculate_f1(cm))

