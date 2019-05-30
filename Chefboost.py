import pandas as pd
import math
import numpy as np
import time
import imp

from commons import functions
from training import Preprocess, Training
from tuning import gbm, adaboost, randomforest

#------------------------

def fit(df, config):
	
	target_label = df.columns[len(df.columns)-1]
	if target_label != 'Decision':
		print("Expected: Decision, Existing: ",target_label)
		raise ValueError('Please confirm that name of the target column is "Decision" and it is put to the right in pandas data frame')
	
	#------------------------
	
	#initialize params and folders
	config = functions.initializeParams(config)
	functions.initializeFolders()
	
	algorithm = config['algorithm']

	enableRandomForest = config['enableRandomForest']
	num_of_trees = config['num_of_trees']
	enableMultitasking = config['enableMultitasking']

	enableGBM = config['enableGBM']
	epochs = config['epochs']
	learning_rate = config['learning_rate']

	enableAdaboost = config['enableAdaboost']
	
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
		for j in range(0, num_of_columns):
			column_name = df.columns[j]
			if df[column_name].dtypes  == 'object':
				raise ValueError('Adaboost must be run on numeric data set for both features and target')
		
	#-------------------------
	
	print(algorithm," tree is going to be built...")
	
	dataset_features = dict() #initialize a dictionary. this is going to be used to check features numeric or nominal. numeric features should be transformed to nominal values based on scales.

	header = "def findDecision("
	header = header + "obj"
	header = header + "): #"
	
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

	if enableAdaboost == True:
		adaboost.apply(df, config, header, dataset_features)

	elif enableGBM == True:
		
		if df['Decision'].dtypes == 'object': #transform classification problem to regression
			gbm.classifier(df, config, header, dataset_features)
			
		else: #regression
			gbm.regressor(df, config, header, dataset_features)
				
	elif enableRandomForest == True:
		randomforest.apply(df, config, header, dataset_features)
	else: #regular decision tree building

		root = 1; file = "outputs/rules/rules.py"
		functions.createFile(file, header)
		Training.buildDecisionTree(df,root,file, config, dataset_features)
	
	print("finished in ",time.time() - begin," seconds")
	
	#-----------------------------------------
