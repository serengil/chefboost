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
	
	#initialize params and folders
	config = functions.initializeParams(config)
	functions.initializeFolders()
	
	debug = config['debug'] 
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
		debug = False #gbm needs rules files to iterate
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

	if(True): #header of rules files
		header = "def findDecision("
		num_of_columns = df.shape[1]-1
		for i in range(0, num_of_columns):
			if debug == True:
				if i > 0:
					header = header + ","
				header = header + df.columns[i]
			
			column_name = df.columns[i]
			dataset_features[column_name] = df[column_name].dtypes

		if debug == False:
			header = header + "obj"
			
		header = header + "):\n"

		if debug == True:
			print(header,end='')	
	
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
		if debug == False: functions.createFile(file, header)
		Training.buildDecisionTree(df,root,file, config, dataset_features)
	
	print("finished in ",time.time() - begin," seconds")
	
	#-----------------------------------------
	
	if config['enableRandomForest'] != True and config['enableGBM'] != True and config['enableAdaboost'] != True:
		#this is reguler decision tree. find accuracy here.
		
		moduleName = "outputs/rules/rules"
		fp, pathname, description = imp.find_module(moduleName)
		myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
		
		num_of_features = df.shape[1] - 1
		instances = df.shape[0]
		classified = 0; mae = 0; mse = 0
		for index, instance in raw_df.iterrows():
			params = []
			for j in range(0, num_of_features):
				params.append(instance[j])
			
			prediction = myrules.findDecision(params)
			actual = instance['Decision']
			
			if algorithm != 'Regression':
				if actual == prediction:
					classified = classified + 1
			else:
				mae = mae + abs(actual - prediction)
				mse = mse + (actual - prediction)*(actual - prediction)
		
		if algorithm != 'Regression':
			accuracy = 100*classified/df.shape[0]
			print("Accuracy: ", accuracy,"% on ",instances," instances")
		else:
			mean = df['Decision'].mean()
			mae = mae / instances
			mse = mse / instances; rmse = math.sqrt(mse)
			
			print("Mean: ",mean)
			print("MAE: ",mae)
			print("RMSE: ",rmse)
			
			if mean > 0:
				print("MAE/Mean: ",100*mae/mean,"%")
				#print("MSE: ",mse)
				#print("MSE/mean: ",100*mse/mean,"%")
				print("RMSE/mean: ",100*rmse/mean,"%")
				print("Instances: ",instances)
