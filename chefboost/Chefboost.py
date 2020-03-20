import pandas as pd
import math
import numpy as np
import time
import imp
import pickle
import os
from os import path

from chefboost.commons import functions
from chefboost.training import Preprocess, Training
from chefboost.tuning import gbm, adaboost, randomforest

#from commons import functions
#from training import Preprocess, Training
#from tuning import gbm, adaboost, randomforest

#------------------------

def fit(df, config):
	
	target_label = df.columns[len(df.columns)-1]
	if target_label != 'Decision':
		print("Expected: Decision, Existing: ",target_label)
		raise ValueError('Please confirm that name of the target column is "Decision" and it is put to the right in pandas data frame')
	
	#------------------------
	#handle NaN values
	
	nan_values = []
	
	for column in df.columns:
		if df[column].dtypes != 'object':
			min_value = df[column].min()
			idx = df[df[column].isna()].index
			
			nan_value = []
			nan_value.append(column)
			
			if idx.shape[0] > 0:
				df.loc[idx, column] = min_value - 1
				nan_value.append(min_value - 1)
				min_value - 1
				#print("NaN values are replaced to ", min_value - 1, " in column ", column)
			else:
				nan_value.append(None)
			
			nan_values.append(nan_value)
	
	#------------------------
	
	#initialize params and folders
	config = functions.initializeParams(config)
	functions.initializeFolders()
	
	#------------------------
	
	algorithm = config['algorithm']
	
	valid_algorithms = ['ID3', 'C4.5', 'CART', 'CHAID', 'Regression']
	
	if algorithm not in valid_algorithms:
		raise ValueError('Invalid algorithm passed. You passed ', algorithm," but valid algorithms are ",valid_algorithms)
	
	#------------------------

	enableRandomForest = config['enableRandomForest']
	num_of_trees = config['num_of_trees']
	enableMultitasking = config['enableMultitasking'] #no longer used. check to remove this variable.

	enableGBM = config['enableGBM']
	epochs = config['epochs']
	learning_rate = config['learning_rate']

	enableAdaboost = config['enableAdaboost']
	enableParallelism = config['enableParallelism']
	
	#this will handle basic decision stumps. parallelism is not required.
	if enableRandomForest == True:
		config['enableParallelism'] = False
		enableParallelism = False
	
	#------------------------
	raw_df = df.copy()
	num_of_rows = df.shape[0]; num_of_columns = df.shape[1]
	
	if algorithm == 'Regression':
		if df['Decision'].dtypes == 'object':
			raise ValueError('Regression trees cannot be applied for nominal target values! You can either change the algorithm or data set.')

	if df['Decision'].dtypes != 'object': #this must be regression tree even if it is not mentioned in algorithm
		algorithm = 'Regression'
		config['algorithm'] = 'Regression'
		global_stdev = df['Decision'].std(ddof=0)

	if enableGBM == True:
		print("Gradient Boosting Machines...")
		algorithm = 'Regression'
		config['algorithm'] = 'Regression'
	
	if enableAdaboost == True:
		#enableParallelism = False
		for j in range(0, num_of_columns):
			column_name = df.columns[j]
			if df[column_name].dtypes  == 'object':
				raise ValueError('Adaboost must be run on numeric data set for both features and target')
		
	#-------------------------
	
	print(algorithm," tree is going to be built...")
	
	dataset_features = dict() #initialize a dictionary. this is going to be used to check features numeric or nominal. numeric features should be transformed to nominal values based on scales.

	header = "def findDecision(obj): #"
	
	num_of_columns = df.shape[1]-1
	for i in range(0, num_of_columns):
		column_name = df.columns[i]
		dataset_features[column_name] = df[column_name].dtypes
		header = header + "obj[" + str(i) +"]: "+column_name
		if i != num_of_columns - 1:
			header = header + ", "
	
	header = header + "\n"
		
	#------------------------
	
	begin = time.time()
	
	trees = []; alphas = []

	if enableAdaboost == True:
		trees, alphas = adaboost.apply(df, config, header, dataset_features)

	elif enableGBM == True:
		
		if df['Decision'].dtypes == 'object': #transform classification problem to regression
			trees, alphas = gbm.classifier(df, config, header, dataset_features)
			classification = True
			
		else: #regression
			trees = gbm.regressor(df, config, header, dataset_features)
			classification = False
				
	elif enableRandomForest == True:
		trees = randomforest.apply(df, config, header, dataset_features)
	else: #regular decision tree building

		root = 1; file = "outputs/rules/rules.py"
		functions.createFile(file, header)
		
		if enableParallelism == True:
			json_file = "outputs/rules/rules.json"
			functions.createFile(json_file, "[\n")
			
		trees = Training.buildDecisionTree(df,root,file, config, dataset_features
			, 0, 0, 'root')
		
	print("finished in ",time.time() - begin," seconds")
	
	obj = {
		"trees": trees,
		"alphas": alphas,
		"config": config,
		"nan_values": nan_values
	}
	
	return obj
	
	#-----------------------------------------

def predict(model, param):
	
	trees = model["trees"]
	config = model["config"]
	alphas = model["alphas"]
	nan_values = model["nan_values"]
	
	#-----------------------
	#handle missing values
	
	column_index = 0
	for column in nan_values:
		column_name = column[0]
		missing_value = column[1]
		
		if pd.isna(missing_value) != True:
			#print("missing values will be replaced with ",missing_value," in ",column_name," column")
			
			if pd.isna(param[column_index]):
				param[column_index] = missing_value
			
		column_index = column_index + 1
			
	#print("instance: ", param)
	#-----------------------
	
	enableGBM = config['enableGBM']
	adaboost = config['enableAdaboost']
	
	#-----------------------
	
	classification = False
	prediction = 0
	prediction_classes = []
	
	#-----------------------
	
	if enableGBM == True:
		
		if len(trees) == config['epochs']:
			classification = False
		else:
			classification = True
			prediction_classes = [0 for i in alphas]
		
	#-----------------------
	
	if len(trees) > 1: #boosting
		index = 0
		for tree in trees:
			if adaboost != True:
				
				custom_prediction = tree.findDecision(param)
				
				if custom_prediction != None:
					if type(custom_prediction) != str: #regression
						
						if enableGBM == True and classification == True:
							prediction_classes[index % len(alphas)] += custom_prediction
						else:
							prediction += custom_prediction
					else:
						classification = True
						prediction_classes.append(custom_prediction)
			else:
				prediction += alphas[index] * tree.findDecision(param)
			index = index + 1
		
		if adaboost == True:
			prediction = functions.sign(prediction)
	else: #regular decision tree
		tree = trees[0]
		prediction = tree.findDecision(param)
	
	if classification == False:
		return prediction
	else:
		if enableGBM == True and classification == True:
			return alphas[np.argmax(prediction_classes)]
		else:
			unique_labels = np.unique(prediction_classes)
			prediction_counts = []
			
			for i in range(0, len(unique_labels)):
				count = 0
				for j in prediction_classes:
					if j == unique_labels[i]:
						count = count + 1
				prediction_counts.append(count)
			
			return unique_labels[np.argmax(prediction_counts)]

def save_model(base_model, file_name="model.pkl"):
	
	model = base_model.copy()
	
	#modules cannot be saved. Save its reference instead.
	module_names = []
	for tree in model["trees"]:
		module_names.append(tree.__name__)

	model["trees"] = module_names
	
	f = open("outputs/rules/"+file_name, "wb")
	pickle.dump(model,f)
	f.close()
	
def load_model(file_name="model.pkl"):
	f = open('outputs/rules/'+file_name, 'rb')
	model = pickle.load(f)
	
	#restore modules from its references
	modules = []
	for model_name in model["trees"]:
		module = functions.restoreTree(model_name)
		modules.append(module)
	
	model["trees"] = modules
	
	return model

def restoreTree(moduleName):
	return functions.restoreTree(moduleName)
