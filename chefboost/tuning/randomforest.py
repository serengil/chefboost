import pandas as pd
import numpy as np

from multiprocessing import Pool

from chefboost.commons import functions
from chefboost.training import Training

from tqdm import tqdm

import imp
	
def apply(df, config, header, dataset_features):
	
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
	#check regression or classification
	if df['Decision'].dtypes == 'object': problem_type = 'classification'
	else: problem_type = 'regression'

	actual_values = df['Decision'].values
	num_of_features = df.shape[1] - 1 #discard Decision
	number_of_instances = df.shape[0]
		
	global_predictions = []
	
	#if classification get the max number of prediction
	if problem_type == 'classification':
		for i in range(0, num_of_trees):
			
			moduleName = "outputs/rules/rule_"+str(i)
			fp, pathname, description = imp.find_module(moduleName)
			myrules = imp.load_module(moduleName, fp, pathname, description)
			
			predictions = []
			
			for index, instance in df.iterrows():
				params = []
				for j in range(0, num_of_features):
					params.append(instance[j])
				
				#index row, i th column 
				prediction = myrules.findDecision(params)
				predictions.append(prediction)
				#print(i,"th tree prediction: ",prediction)
					
			#print(predictions)
			global_predictions.append(predictions)
			
		#-------------------------------
		classified = 0
		for index, instance in df.iterrows():
			
			actual = actual_values[index]
			predictions = []
			for i in range(0, num_of_trees):
				prediction = global_predictions[i][index]
				if prediction != None: #why None exists in some cases?
					predictions.append(prediction)
				
			predictions = np.array(predictions)
			unique_values = np.unique(predictions)
			
			if unique_values.shape[0] == 1:
				prediction = unique_values[0]
			else:
				counts = []
				for unique in unique_values:
					count = 0
					for j in predictions:
						if unique == j:
							count = count + 1
					counts.append(count)
				
				#print("unique: ",unique_values)
				#print("counts: ",counts)
				
				prediction = None
				
				if len(counts) > 0:
					max_index = np.argmax(np.array(counts))
					prediction = unique_values[max_index]
						
			#print(index,". actual: ",actual," - prediction: ", prediction)
			if actual == prediction:
				classified = classified + 1
		
		print("Accuracy: ",100*classified/number_of_instances,"% on ",number_of_instances," instances")
	
	return models
