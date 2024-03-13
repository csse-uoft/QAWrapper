# -*- coding: utf-8 -*-
"""qa_layer1_eval.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ALSOoF4Do_W8XscyBuN2TUjuF-gu9WLS

# 1. Setup
* import libraries
* change directory
* annotation maps (model labels $→$ doccano labels)
* stopword list
"""

from google.colab import drive
drive.mount('/content/drive')

import os, json, copy
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from google.colab import drive
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import itertools as itert
# !pip install --upgrade xlrd
import xlrd

# Annotation map

def hs_annotations_parse(correct=True):
  HS_entity_map = {'Outcome': 'need/outcome',
                  'Client': 'client',
                  'Need Satisfier': 'need satisfier',
                  'Program': "program name",
                  'CatchmentArea': "catchment area",
                  'BeneficialStakeholder': "client",
                  'Eligibility': "client",
                  'Service Mode': None,
                  'Ignore': None,
                  'Service': 'need satisfier',
                  'Need': "need/outcome",
                  'ContributingStakeholder': None,
                  'Requirement': None,
                  'Community': "client",
                  'Availability': None,
                  'Organization': "program name",
                  'Facility': None}
  filename = "drive/MyDrive/NLP Project/data/HS_June5_all_corrected.xlsx"
  xls = pd.ExcelFile(filename)
  sn = xls.sheet_names
  all_HS_dfs = {}

  for sheet in sn:
    try:
      all_HS_dfs[sheet] = pd.read_excel(filename, sheet_name = sheet)[["idx", "extracted term", "start", "end", "full text"]]
    except KeyError:
      continue

  # all_HS_dfs = hs_correct(all_HS_dfs)


  all_true_dfs = {"program name": pd.DataFrame(columns=["idx", "extracted term", "start", "end", "full text"]),
                  "client": pd.DataFrame(columns=["idx", "extracted term", "start", "end", "full text"]),
                  "need satisfier": pd.DataFrame(columns=["idx", "extracted term", "start", "end", "full text"]),
                  "need/outcome": pd.DataFrame(columns=["idx", "extracted term", "start", "end", "full text"]),
                  "catchment area": pd.DataFrame(columns=["idx", "extracted term", "start", "end", "full text"])}
  for HS_entity in all_HS_dfs:
    try:
      if HS_entity_map[HS_entity] is not None:
        this_df = all_HS_dfs[HS_entity]
        ent = HS_entity_map[HS_entity]
        all_true_dfs[ent] = pd.concat([all_true_dfs[ent], this_df], ignore_index=True)
    except KeyError:
      continue
  return all_true_dfs

def hs_correct(all_HS_dfs):

  pr_corrected = "drive/MyDrive/NLP Project/data/HS-program-corrected.xls"
  df_pr = pd.read_excel(pr_corrected, sheet_name = "HS program corrected")[["idx", "extracted term", "start", "end", "corrected", "full text"]]
  sv_corrected = "drive/MyDrive/NLP Project/data/HS-service-corrected.xls"
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

all_true_dfs = hs_annotations_parse()

def hs_layer1_result_parse():
  f1 = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/pr_layer1_res_HS.xlsx"
  f2 = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/cl_layer1_res_HS.xlsx"
  f3 = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/ns_layer1_res_HS.xlsx"
  f4 = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/no_layer1_res_HS.xlsx"
  f5 = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/ca_layer1_res_HS.xlsx"

  all_res_dfs = {"program name": [],
                "client": [],
                "need satisfier": [],
                "need/outcome": [],
                "catchment area": []}

  undo_abb = {"program": "program name", "client": "client", "need satisfier": "need satisfier", "need/outcome": "need/outcome", "catchment area": "catchment area"}

  for f in [f1, f2, f3, f4, f5]:
    this_df = pd.read_excel(f)
    for i, row in this_df.iterrows():
      ent = undo_abb[row.at["res entity"]]
      all_res_dfs[ent].append(row)

  for ent in all_res_dfs:
    all_res_dfs[ent] = pd.DataFrame(all_res_dfs[ent])
    all_res_dfs[ent] = all_res_dfs[ent].drop(['Unnamed: 0'], axis=1)

  return all_res_dfs

def hs_new_layer1_result_parse():
  f = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/rank_results/HS_layer1_fixed_answers.csv"

  all_res_dfs = {"program name": [],
                 "client": [],
                 "need satisfier": [],
                 "need/outcome": [],
                 "catchment area": []}

  undo_abb = {"program name": "program name", "client": "client", "need satisfier": "need satisfier", "need outcome": "need/outcome", "catchment area": "catchment area"}

  this_df = pd.read_csv(f)
  for i, row in this_df.iterrows():
    ent = undo_abb[row.at["target entity"]]
    all_res_dfs[ent].append(row)

  for ent in all_res_dfs:
    all_res_dfs[ent] = pd.DataFrame(all_res_dfs[ent])

  return all_res_dfs

def hs_preannotated_result_parse():
  f = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/HS_qa_preannotated.xlsx"
  undo_abb = {"program name": "program name", "client": "client", "need satisfier": "need satisfier", "need outcome": "need/outcome", "catchment area": "catchment area"}
  xls = pd.ExcelFile(f)
  sn = xls.sheet_names
  all_true_dfs = {"program name": pd.DataFrame(columns=["idx", "qid", "response", "eval", "score", "start", "end", "text"]),
                  "client": pd.DataFrame(columns=["idx", "qid", "response", "eval", "score", "start", "end", "text"]),
                  "need satisfier": pd.DataFrame(columns=["idx", "qid", "response", "eval", "score", "start", "end", "text"]),
                  "need/outcome": pd.DataFrame(columns=["idx", "qid", "response", "eval", "score", "start", "end", "text"]),
                  "catchment area": pd.DataFrame(columns=["idx", "qid", "response", "eval", "score", "start", "end", "text"])}
  for sheet in sn:
    this_df = pd.read_excel(f, sheet_name = sheet)[["idx", "qid", "response", "eval", "score", "start", "end", "text"]]
    ent = undo_abb[sheet]
    all_true_dfs[ent] = pd.concat([all_true_dfs[ent], this_df], ignore_index=True)
  return all_true_dfs

with open('drive/MyDrive/NLP Project/data/QA_results/HS June/cleaned_HS_description.json') as json_file:
    HS_des_dic = json.load(json_file)
HS_des_df = pd.DataFrame.from_dict(HS_des_dic, orient="index", columns=['description'])

# Stopwords (Y5)
STOPWORDS = ["it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", ",", 
             "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
             "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", 
             "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
             "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
             "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don't", "should", "now", "'"]

"""#2. Functions"""

nltk_stopwords = stopwords.words('english')
def flatten(l):
    return list(itert.chain.from_iterable(l))

def get_common_terms(df, col='description', quantile=.95):
  terms = [word_tokenize(ts) for ts in df[col].values]
  terms = flatten(terms)
  terms = [t.lower() for t in terms if t.lower() not in nltk_stopwords + ['child','children','family','families','information','parent','parents','(',')','.',':',',','`','’','!','\'s' '\'', '\"', ';','&', '-', 'amp']]
  sr = pd.Series(terms)
  tmp = sr.value_counts()
  threshold = tmp.quantile(quantile)
  return tmp[tmp>threshold].index.tolist()

COMMON_TERMS = get_common_terms(HS_des_df)
COMMON_TERMS[:20]

# def overlap(x,y):
#   '''
#   return overlapping section of string paramters x and y
#   '''
#   d = difflib.SequenceMatcher(None,x,y)
#   match = max(d.get_matching_blocks(),key=lambda x:x[2])
#   i,j,k = match
#   return d.a[i:i+k]
def overlap(x,y):
  return set(x).intersection(y)

def set_eval_y(term, cur_term):
    evals = {}
    overlapped = overlap(word_tokenize(cur_term), word_tokenize(term))
    evals["Y1"] = (term == cur_term)
    evals["Y2"] = (cur_term in term)
    evals["Y3"] = (len([t for t in overlapped if t not in COMMON_TERMS + STOPWORDS]) > 0)
    if evals["Y3"]:
      evals["Y4"] = False
    else:
      evals["Y4"] = (len(set([t for t in overlapped if t not in STOPWORDS]).intersection(COMMON_TERMS))>0)# not in COMMON_TERMS)
    if evals["Y4"]:
      evals["Y5"] = False
    else:
      evals["Y5"] = not evals["Y3"] and (len(set([t for t in overlapped if t not in COMMON_TERMS]).intersection(STOPWORDS))>0)# not in COMMON_TERMS)
    return evals

def main_evaluate(cand_terms, term):
  """Evaluate the model output term given candidates of terms, without using any indexing information.

  Args:
    cand_terms (list): doccano annotated terms that are from the same sample and fall under the right categories of interest
    term (str): a single NE from the model that needs to be evaluated
  
  Returns:
    str: evaluation of the term [Y1, Y2, Y3, Y4, Y5, N]

  """
  evals_list = []
  for cur_term in cand_terms:
    evals = set_eval_y(term, cur_term)
    evals['truth'] = cur_term
    evals_list.append(evals)

  
  # tmp = pd.DataFrame(evals_list)#.sort_values(['Y1', 'Y2', 'Y3', 'Y4', 'Y5'],ascending=False)
  # evals = pd.DataFrame(tmp).max().to_dict()
  res = []
  for evals in evals_list:
    keep_eval = 'N'
    for eval,val in evals.items():
      if type(val) is bool and val:
        keep_eval = eval
        break
    if keep_eval != 'N':
      res.append({'eval':keep_eval, 'truth':evals['truth']})
  if len(res)>0:
    return res
  else:
    return [{'eval':'N', 'truth':None}]

def range_overlap(extracted_indices, cand_indices):
  start1, end1 = extracted_indices
  start2, end2 = cand_indices
  """Does the range (start1, end1) overlap with (start2, end2)?"""
  return end1 >= start2 and end2 >= start1

def evaluate(true_dfs, term, extracted_indices=None):
  """Final evaluation of one row of the NER model outputs.

  Args:
  
  Returns:
    list: evaluation of the term [Y1, Y2, Y3, Y4, (Y5), N] and the true term; as a list of hashes
          [{'eval':<eval term>, 'truth':<true value>}]

  """
  term = term.strip()
  cand_terms = []
  indices = []
  for true_df in true_dfs:
    cand_terms += [true_df.iloc[i].at["extracted term"].strip() for i in range(len(true_df))]
    indices += [[true_df.iloc[i].at["start"], true_df.iloc[i].at["end"]] for i in range(len(true_df))]

  main_evals = main_evaluate(cand_terms, term)
  final_evals = []
  for main_eval in main_evals:
    if main_eval['eval'] == "N" or extracted_indices is None:
      final_evals.append(main_eval)
    else:
      for i in np.where(np.array(cand_terms) == main_eval['truth'].strip())[0]:
        if range_overlap(extracted_indices, indices[i]):
          final_evals.append(main_eval)

  y1_final_evals = [eval for eval in final_evals if eval['eval']=='Y1']
  if len(y1_final_evals)>0:
    return y1_final_evals
  else:
    return final_evals

def all_evaluate(all_true_dfs, all_model_dfs, new_xlsx_path=None):

  for ent in all_true_dfs:
    true_df = all_true_dfs[ent]
    model_df = all_model_dfs[ent] 
    model_df["eval"] = ""
    res = []
    for i, row in model_df.iterrows():
      idx = row.at["idx"]
      term = row.at["response"]

      try:
        extracted_indices = [row.at["start"], row.at["end"]]
        if extracted_indices[0] != extracted_indices[0] or extracted_indices[0] != extracted_indices[0]:
          extracted_indices = None  
      except KeyError:
        extracted_indices = None

      cor_true_dfs = [true_df[true_df["idx"] == idx]]
      evaluations = evaluate(cor_true_dfs, term, extracted_indices)
      tmp = row.copy()
      for evaluation in evaluations:
        tmp["eval"] = evaluation['eval']
        tmp["truth"] = evaluation['truth']
        res.append(tmp.copy())
    all_model_dfs[ent] = pd.DataFrame(res).drop_duplicates()

  if new_xlsx_path is not None:
    with ExcelWriter(new_xlsx_path) as writer:
      for entity in all_model_dfs:
        e = entity.replace("/", " ")
        all_model_dfs[entity].to_excel(writer, sheet_name=e, index=False)

  return all_model_dfs

all_true_dfs = hs_annotations_parse()
# all_res_dfs = hs_layer1_result_parse()
all_res_dfs = hs_new_layer1_result_parse()
# all_res_dfs = hs_preannotated_result_parse()

xlsx_path = "drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/rank_results/HS_layer1_fixed_eval.xlsx"
eval_df = all_evaluate(all_true_dfs, all_res_dfs, new_xlsx_path=xlsx_path)

eval_df["program name"]

eval_df["note"] = ""
sampled_df = eval_df.sample(n=200, random_state=2)

with ExcelWriter("drive/MyDrive/NLP Project/data/QA_results/QA layer/answers/HS June/HS_layer1_eval2.xlsx") as writer:
  eval_df.to_excel(writer, index=False)