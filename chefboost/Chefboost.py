import pandas as pd
import math
import numpy as np
import time
import imp
import pickle
import os
from os import path
import json

from chefboost.commons import functions, evaluate as eval
from chefboost.training import Preprocess, Training
from chefboost.tuning import gbm, adaboost, randomforest

#------------------------

def fit(df, config = {}, target_label = 'Decision', validation_df = None):

	"""
	Parameters:
		df (pandas data frame): Training data frame. The target column must be named as 'Decision' and it has to be in the last column

		config (dictionary):

			config = {
				'algorithm' (string): ID3, 'C4.5, CART, CHAID or Regression
				'enableParallelism' (boolean): False

				'enableGBM' (boolean): True,
				'epochs' (int): 7,
				'learning_rate' (int): 1,

				'enableRandomForest' (boolean): True,
				'num_of_trees' (int): 5,

				'enableAdaboost' (boolean): True,
				'num_of_weak_classifier' (int): 4
			}

		validation_df (pandas data frame): if nothing is passed to validation data frame, then the function validates built trees for training data frame

	Returns:
		chefboost model

	"""

	#------------------------

	process_id = os.getpid()

	#------------------------
	#rename target column name
	if target_label != 'Decision':
		df = df.rename(columns = {target_label: 'Decision'})

	#if target is not the last column
	if df.columns[-1] != 'Decision':
		if 'Decision' in df.columns:
			new_column_order = df.columns.drop('Decision').tolist() + ['Decision']
			#print(new_column_order)
			df = df[new_column_order]
		else:
			raise ValueError('Please set the target_label')

	#------------------------

	base_df = df.copy()

	#------------------------

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

	#------------------------

	if enableParallelism == True:
		print("[INFO]: ",config["num_cores"],"CPU cores will be allocated in parallel running")

		from multiprocessing import set_start_method, freeze_support
		set_start_method("spawn", force=True)
		freeze_support()
	#------------------------
	raw_df = df.copy()
	num_of_rows = df.shape[0]; num_of_columns = df.shape[1]

	if algorithm == 'Regression':
		if df['Decision'].dtypes == 'object':
			raise ValueError('Regression trees cannot be applied for nominal target values! You can either change the algorithm or data set.')

	if df['Decision'].dtypes != 'object': #this must be regression tree even if it is not mentioned in algorithm

		if algorithm != 'Regression':
			print("WARNING: You set the algorithm to ", algorithm," but the Decision column of your data set has non-object type.")
			print("That's why, the algorithm is set to Regression to handle the data set.")

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
		trees, alphas = adaboost.apply(df, config, header, dataset_features, validation_df = validation_df, process_id = process_id)

	elif enableGBM == True:

		if df['Decision'].dtypes == 'object': #transform classification problem to regression
			trees, alphas = gbm.classifier(df, config, header, dataset_features, validation_df = validation_df, process_id = process_id)
			classification = True

		else: #regression
			trees = gbm.regressor(df, config, header, dataset_features, validation_df = validation_df, process_id = process_id)
			classification = False

	elif enableRandomForest == True:
		trees = randomforest.apply(df, config, header, dataset_features, validation_df = validation_df, process_id = process_id)
	else: #regular decision tree building

		root = 1; file = "outputs/rules/rules.py"
		functions.createFile(file, header)

		if enableParallelism == True:
			json_file = "outputs/rules/rules.json"
			functions.createFile(json_file, "[\n")

		trees = Training.buildDecisionTree(df, root = root, file = file, config = config
				, dataset_features = dataset_features
				, parent_level = 0, leaf_id = 0, parents = 'root', validation_df = validation_df, main_process_id = process_id)

	print("-------------------------")
	print("finished in ",time.time() - begin," seconds")

	obj = {
		"trees": trees,
		"alphas": alphas,
		"config": config,
		"nan_values": nan_values
	}

	#-----------------------------------------

	#train set accuracy
	df = base_df.copy()
	evaluate(obj, df, task = 'train')

	#validation set accuracy
	if isinstance(validation_df, pd.DataFrame):
		evaluate(obj, validation_df, task = 'validation')

	#-----------------------------------------

	return obj

	#-----------------------------------------

def predict(model, param):

	"""
	Parameters:
		model (built chefboost model): you should pass model argument to the return of fit function
		param (list): pass input features as python list

		e.g. chef.predict(model, param = ['Sunny', 'Hot', 'High', 'Weak'])
	Returns:
		prediction
	"""

	trees = model["trees"]
	config = model["config"]

	alphas = []
	if "alphas" in model:
		alphas = model["alphas"]

	nan_values = []
	if "nan_values" in model:
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
	enableRandomForest = config['enableRandomForest']

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

	if len(trees) > 1: #bagging or boosting
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
			else: #adaboost
				prediction += alphas[index] * tree.findDecision(param)
			index = index + 1

		if enableRandomForest == True:
			#notice that gbm requires cumilative sum but random forest requires mean of each tree
			prediction = prediction / len(trees)

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
		else: #classification
			#e.g. random forest
			#get predictions made by different trees
			predictions = np.array(prediction_classes)

			#find the most frequent prediction
			(values, counts) = np.unique(predictions, return_counts=True)
			idx = np.argmax(counts)
			prediction = values[idx]

			return prediction

def save_model(base_model, file_name="model.pkl"):

	"""
	Parameters:
		base_model (built chefboost model): you should pass this to the return of fit function
		file_name (string): you should pass target file name as exact path.
	"""

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

	"""
	Parameters:
		file_name (string): exact path of the target saved model
	Returns:
		built chefboost model
	"""

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

	"""
	If you have decision rules, then this function enables you to load a built chefboost model. You can then call prediction.
	Parameters:
		moduleName (string): you should pass outputs/rules/rules if you want to restore outputs/rules/rules.py

	Returns:
		built chefboost model
	"""

	return functions.restoreTree(moduleName)

def feature_importance(rules):

	"""
	Parameters:
		rules (string or list):

		e.g. decision_rules = "outputs/rules/rules.py"
		or this could be retrieved from built model as shown below.

			decision_rules = []
			for tree in model["trees"]:
			   rule = .__dict__["__spec__"].origin
			   decision_rules.append(rule)

	Returns:
		pandas data frame
	"""

	if type(rules) != list:
		rules = [rules]
	else:
		print("rules: ",rules)

	#-----------------------------

	dfs = []

	for rule in rules:
		print("Decision rule: ",rule)

		file = open(rule, 'r')
		lines = file.readlines()

		pivot = {}
		rules = []

		#initialize feature importances
		line_idx = 0
		for line in lines:
			if line_idx == 0:
				feature_explainer_list = line.split("#")[1].split(", ")
				for feature_explainer in feature_explainer_list:
					feature = feature_explainer.split(": ")[1].replace("\n", "")
					pivot[feature] = 0
			else:
				if "# " in line:
					rule = line.strip().split("# ")[1]
					rules.append(json.loads(rule))

			line_idx = line_idx + 1

		feature_names = list(pivot.keys())

		for feature in feature_names:
			for rule in rules:
				if rule["feature"] == feature:


					score = rule["metric_value"] * rule["instances"]
					current_depth = rule["depth"]

					child_scores = 0
					#find child node importances
					for child_rule in rules:
						if child_rule["depth"] == current_depth + 1:

							child_score = child_rule["metric_value"] * child_rule["instances"]

							child_scores = child_scores + child_score

					score = score - child_scores

					pivot[feature] = pivot[feature] + score

		#normalize feature importance

		total_score = 0
		for feature, score in pivot.items():
			total_score = total_score + score

		for feature, score in pivot.items():
			pivot[feature] = round(pivot[feature] / total_score, 4)

		instances = []
		for feature, score in pivot.items():
			instance = []
			instance.append(feature)
			instance.append(score)
			instances.append(instance)

		df = pd.DataFrame(instances, columns = ["feature", "final_importance"])
		df = df.sort_values(by = ["final_importance"], ascending = False)

		if len(rules) == 1:
			return df
		else:
			dfs.append(df)

	if len(rules) > 1:

		hf = pd.DataFrame(feature_names, columns = ["feature"])
		hf["importance"] = 0

		for df in dfs:
			hf = hf.merge(df, on = ["feature"], how = "left")
			hf["importance"] = hf["importance"] + hf["final_importance"]
			hf = hf.drop(columns = ["final_importance"])

		#------------------------
		#normalize
		hf["importance"] = hf["importance"] / hf["importance"].sum()
		hf = hf.sort_values(by = ["importance"], ascending = False)

		return hf

def evaluate(model, df, target_label = 'Decision', task = 'test'):

	"""
	Parameters:
		model (built chefboost model): you should pass the return of fit function
		df (pandas data frame): data frame you would like to evaluate
		task (string): optionally you can pass this train, validation or test
	"""

	#--------------------------

	if target_label != 'Decision':
		df = df.rename(columns = {target_label: 'Decision'})

	#if target is not the last column
	if df.columns[-1] != 'Decision':
		new_column_order = df.columns.drop('Decision').tolist() + ['Decision']
		print(new_column_order)
		df = df[new_column_order]

	#--------------------------

	functions.bulk_prediction(df, model)

	enableAdaboost = model["config"]["enableAdaboost"]

	if enableAdaboost == True:
		df['Decision'] = df['Decision'].astype(str)
		df['Prediction'] = df['Prediction'].astype(str)

	eval.evaluate(df, task = task)
