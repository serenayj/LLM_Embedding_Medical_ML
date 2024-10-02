import os
#os.environ['CUDA_VISIBLE_DEVICES']="1,2"

import torch
from openai.embeddings_utils import get_embedding # only exists on openai==0.28.1 
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import gc
import pandas as pd
import numpy as np 
from tab_embed_utils import *
from xgb_utils import *
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn import metrics
import argparse

import json
config_file="model_path.json" # this is the json file storing paths to model checkpoints
with open(config_file,"r") as file:
	model_path_config = json.load(file)['model_path']


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str,required=True,help="Specify embedding model")
	parser.add_argument("--embedding_method",type=str,default="mean_pooling",required=True, help="Specify embedding method")
	parser.add_argument("--harness",action="store_true",help="If true, use binary question in the prompt")
	parser.add_argument("--target_diagnosis",type=str,default="sepsis",help="Specify the target diagnosis you want to predict")
	parser.add_argument("--site",type=str,default="UW",help="Specify which site you want to use") 

	args = parser.parse_args()

	if "llama2-7b" in args.model_name.lower():
		model_type = "llama2-7b-chat" 
	elif "llama2-13b" in args.model_name.lower():
		model_type = "llama2-13b-chat" 
	elif "mistral" in args.model_name.lower():
		model_type = "mistral-7b-instruct"
	elif "bge"in args.model_name:
		model_type = "bge"
	elif "meditron" in args.model_name:
		model_type = "meditron-7b"
	elif "clinicalbert" in args.model_name.lower():
		model_type = "clinicalBERT"
	else:
		raise NotImplementedError

	method = args.embedding_method

	emb_model = EmbeddingModel(model_type, method, model_path_config)

	emb_model.load()
	print(f"== {model_type} Model Has Been Loaded ==")

	if args.site == "UW":
		if args.harness:
			data=pd.read_csv(f"llm_xgb_input/llm+xgb_data_{args.target_diagnosis}_harness.csv")
			exp_name = model_type+"_"+args.embedding_method+"_"+args.target_diagnosis+"_harness"
		else:
			data = pd.read_csv(f"llm_xgb_input/llm+xgb_data_{args.target_diagnosis}.csv")
			exp_name = model_type+"_"+args.embedding_method+"_"+args.target_diagnosis
	elif args.site == "UC":
		if args.harness:
			data=pd.read_csv(f"UC_data/llm+xgb_data_{args.target_diagnosis}_harness.csv")
			exp_name = args.site+"_"+model_type+"_"+args.embedding_method+"_"+args.target_diagnosis+"_harness"
		else:
			data = pd.read_csv(f"UC_data/llm+xgb_data_{args.target_diagnosis}.csv")
			exp_name = args.site+"_"+model_type+"_"+args.embedding_method+"_"+args.target_diagnosis
	else:
		print(f"Site {args.site} Not Found!!!!")
		raise NotImplementedError 

	print(f"***** EXP SETUP: {exp_name} ***** ")
	print(f"Make sure the exp setup is correct before proceeding ")

	test_enc_ids, batch_auroc, batch_yproba, batch_y_labels = main_kfold_xgboost(emb_model, data, args.site)

	import pickle

	output_dict = {"Y_Proba": batch_yproba,"Y_Labels": batch_y_labels, "batch_auroc":batch_auroc,"batch_ids": test_enc_ids}

	draw_curve(batch_y_labels, batch_yproba, exp_name)

	with open("Output/"+exp_name+".pkl","wb") as outf:
		pickle.dump(output_dict, outf)

	


