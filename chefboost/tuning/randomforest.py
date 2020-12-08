import pandas as pd
import numpy as np

from multiprocessing import Pool

from chefboost.commons import functions, evaluate
from chefboost.training import Training
from chefboost import Chefboost as cb

from tqdm import tqdm

import imp
	
def apply(df, config, header, dataset_features, validation_df = None):
	
	models = []
	
	num_of_trees = config['num_of_trees']
	
	pbar = tqdm(range(0, num_of_trees), desc='Bagging')
	
	for i in pbar:
	#for i in range(0, num_of_trees):
		pbar.set_description("Sub decision tree %d is processing" % (i+1))
		subset = df.sample(frac=1/num_of_trees)
		
		root = 1
		
		moduleName = "outputs/rules/rule_"+str(i)
		file = moduleName+".py"; json_file = moduleName+".json"
		
		functions.createFile(file, header)
		functions.createFile(json_file, "[\n")
		
		Training.buildDecisionTree(subset,root, file, config, dataset_features
			, parent_level = 0, leaf_id = 0, parents = 'root')
		
		functions.storeRule(json_file,"{}]")
		
		#--------------------------------
		
		fp, pathname, description = imp.find_module(moduleName)
		myrules = imp.load_module(moduleName, fp, pathname, description)
		models.append(myrules)
		
	#-------------------------------
	
	return models