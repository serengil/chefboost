import math
import imp
import uuid
import json
import numpy as np
import copy

from chefboost.training import Preprocess
from chefboost.commons import functions

def listToString(my_list):
	my_text = "["
	for i in range(0, len(my_list)):
		my_text += "\""+my_list[i]+"\""
		if i < len(my_list) - 1:
			my_text += ", "
	my_text += "]"
	return my_text

def calculateEntropy(df, config):
	
	algorithm = config['algorithm']
	
	#--------------------------
	
	if algorithm == 'Regression':
		return 0
	
	#print(df)

	instances = df.shape[0]; columns = df.shape[1]
	#print(instances," rows, ",columns," columns")

	decisions = df['Decision'].value_counts().keys().tolist()

	entropy = 0

	for i in range(0, len(decisions)):
		decision = decisions[i]
		num_of_decisions = df['Decision'].value_counts().tolist()[i]
		#print(decision,"->",num_of_decisions)
		
		class_probability = num_of_decisions/instances
		
		entropy = entropy - class_probability*math.log(class_probability, 2)
		
	return entropy

def findDecision(df, config):
	
	algorithm = config['algorithm']
	
	#-----------------------------
	
	if algorithm == 'Regression':
		stdev = df['Decision'].std(ddof=0)
		
	
	entropy = 0
	
	if algorithm == "ID3" or algorithm == "C4.5":
		entropy = calculateEntropy(df, config)
	#print("entropy: ",entropy)

	columns = df.shape[1]; instances = df.shape[0]

	gains = []; gainratios = []; ginis = []; reducted_stdevs = []

	for i in range(0, columns-1):
		column_name = df.columns[i]
		column_type = df[column_name].dtypes
		
		#print(column_name,"->",column_type)
		
		if column_type != 'object':
			df = Preprocess.processContinuousFeatures(algorithm, df, column_name, entropy, config)
		
		classes = df[column_name].value_counts()
		
		gain = entropy * 1; splitinfo = 0; gini = 0; weighted_stdev = 0
		
		for j in range(0, len(classes)):
			current_class = classes.keys().tolist()[j]
			#print(column_name,"->",current_class)
			
			subdataset = df[df[column_name] == current_class]
			#print(subdataset)
			
			subset_instances = subdataset.shape[0]
			class_probability = subset_instances/instances
			
			if algorithm == 'ID3' or algorithm == 'C4.5':
				subset_entropy = calculateEntropy(subdataset, config)
				#print("entropy for this sub dataset is ", subset_entropy)
				gain = gain - class_probability * subset_entropy			
			
			if algorithm == 'C4.5':
				splitinfo = splitinfo - class_probability*math.log(class_probability, 2)
			
			elif algorithm == 'CART': #GINI index
				decision_list = subdataset['Decision'].value_counts().tolist()
				
				subgini = 1
				
				for k in range(0, len(decision_list)):
					subgini = subgini - math.pow((decision_list[k]/subset_instances), 2)
				
				gini = gini + (subset_instances / instances) * subgini
			
			elif algorithm == 'Regression':
				subset_stdev = subdataset['Decision'].std(ddof=0)
				weighted_stdev = weighted_stdev + (subset_instances/instances)*subset_stdev
		
		#iterating over classes for loop end
		#-------------------------------
		
		if algorithm == "ID3":
			gains.append(gain)
		
		elif algorithm == "C4.5":
		
			if splitinfo == 0:
				splitinfo = 100 #this can be if data set consists of 2 rows and current column consists of 1 class. still decision can be made (decisions for these 2 rows same). set splitinfo to very large value to make gain ratio very small. in this way, we won't find this column as the most dominant one.
				
			gainratio = gain / splitinfo
			gainratios.append(gainratio)
		
		elif algorithm == "CART":
			ginis.append(gini)
		
		elif algorithm == 'Regression':
			reducted_stdev = stdev - weighted_stdev
			reducted_stdevs.append(reducted_stdev)
	
	#print(df)
	if algorithm == "ID3":
		winner_index = gains.index(max(gains))
	elif algorithm == "C4.5":
		winner_index = gainratios.index(max(gainratios))
	elif algorithm == "CART":
		winner_index = ginis.index(min(ginis))
	elif algorithm == "Regression":
		winner_index = reducted_stdevs.index(max(reducted_stdevs))
	winner_name = df.columns[winner_index]

	return winner_name

def createBranch():
	return 0

def buildDecisionTree(df, root, file, config, dataset_features, parent_level = 0, leaf_id = 0, parents = 'root'):
	
	json_file = file.split(".")[0]+".json"
	
	models = []

	if root == 1:
		if config['enableRandomForest'] != True and config['enableGBM'] != True and config['enableAdaboost'] != True:
			raw_df = df.copy()
	
	algorithm = config['algorithm']
	enableAdaboost = config['enableAdaboost']
	enableParallelism = config['enableParallelism']
	
	#--------------------------------------
	
	charForResp = "'"
	if algorithm == 'Regression':
		charForResp = ""

	tmp_root = root * 1
	parents_raw = copy.copy(parents)
	
	df_copy = df.copy()
	
	winner_name = findDecision(df, config)
	
	#find winner index, this cannot be returned by find decision because columns dropped in previous steps
	j = 0 
	for i in dataset_features:
		if i == winner_name:
			winner_index = j
		j = j + 1
	
	numericColumn = False
	if dataset_features[winner_name] != 'object':
		numericColumn = True
	
	#restoration
	columns = df.shape[1]
	for i in range(0, columns-1):
		column_name = df.columns[i]; column_type = df[column_name].dtypes
		if column_type != 'object' and column_name != winner_name:
			df[column_name] = df_copy[column_name]
	
	classes = df[winner_name].value_counts().keys().tolist()
		
	#-----------------------------------------------------
	
	#TO-DO: This block could be paralellised
	
	#-----------------------------------------------------
	
	#serial approach
	for i in range(0,len(classes)):
		current_class = classes[i]
		subdataset = df[df[winner_name] == current_class]
		subdataset = subdataset.drop(columns=[winner_name])
		
		if numericColumn == True:
			compareTo = current_class #current class might be <=x or >x in this case
		else:
			compareTo = " == '"+str(current_class)+"'"
		
		#print(subdataset)
		
		terminateBuilding = False
		
		#-----------------------------------------------
		#can decision be made?
		
		if enableAdaboost == True:
			#final_decision = subdataset['Decision'].value_counts().idxmax()
			final_decision = functions.sign(subdataset['Decision'].mean()) #get average
			terminateBuilding = True
			enableParallelism = False
		elif len(subdataset['Decision'].value_counts().tolist()) == 1:
			final_decision = subdataset['Decision'].value_counts().keys().tolist()[0] #all items are equal in this case
			terminateBuilding = True
		elif subdataset.shape[1] == 1: #if decision cannot be made even though all columns dropped
			final_decision = subdataset['Decision'].value_counts().idxmax() #get the most frequent one
			terminateBuilding = True
		elif algorithm == 'Regression' and subdataset.shape[0] < 5: #pruning condition
		#elif algorithm == 'Regression' and subdataset['Decision'].std(ddof=0)/global_stdev < 0.4: #pruning condition
			final_decision = subdataset['Decision'].mean() #get average
			terminateBuilding = True
		#-----------------------------------------------
		
		if i == 0:
			check_condition = "if"
		else:
			check_condition = "elif"
		
		check_rule = check_condition+" obj["+str(winner_index)+"]"+compareTo+":"
		
		if enableParallelism != True:
			functions.storeRule(file,(functions.formatRule(root),"",check_rule))
		else:
			leaf_id = str(uuid.uuid1())
			
			sample_rule = "   {\n"
			sample_rule += "      \"current_level\": "+str(root)+",\n"
			sample_rule += "      \"leaf_id\": \""+str(leaf_id)+"\",\n"
			sample_rule += "      \"parents\": \""+parents+"\",\n"
			sample_rule += "      \"rule\": \""+check_rule+"\"\n"
			sample_rule += "   },"

			functions.storeRule(json_file, sample_rule)
		
		#-----------------------------------------------
		
		if terminateBuilding == True: #check decision is made
			
			parents = copy.copy(leaf_id)
			leaf_id = str(uuid.uuid1())
			
			decision_rule = "return "+charForResp+str(final_decision)+charForResp
			
			if enableParallelism != True:
				#serial
				functions.storeRule(file,(functions.formatRule(root+1),decision_rule))
			else:
				#parallel
				sample_rule = "   {\n"
				sample_rule += "      \"current_level\": "+str(root+1)+",\n"
				sample_rule += "      \"leaf_id\": \""+str(leaf_id)+"\",\n"
				sample_rule += "      \"parents\": \""+parents+"\",\n"
				sample_rule += "      \"rule\": \""+decision_rule+"\"\n"
				sample_rule += "   }, "
				
				functions.storeRule(json_file, sample_rule)
		
		else: #decision is not made, continue to create branch and leafs
			root = root + 1 #the following rule will be included by this rule. increase root
			
			parents = copy.copy(leaf_id)
			
			buildDecisionTree(subdataset, root, file, config, dataset_features
				, root-1, leaf_id, parents)
						
		root = tmp_root * 1
		parents = copy.copy(parents_raw)
	
	#---------------------------------------------
	
	#calculate accuracy metrics
	if root == 1:
		
		if enableParallelism == True:
			file_name = file.split(".py")[0].split("/")[2]
			functions.storeRule(json_file, "{}]")
			reconstructRules(file_name+".json")
		
		if config['enableRandomForest'] != True and config['enableGBM'] != True and config['enableAdaboost'] != True:
		#this is reguler decision tree. find accuracy here.
			
			moduleName = "outputs/rules/rules"
			fp, pathname, description = imp.find_module(moduleName)
			myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
			models.append(myrules)
			
			num_of_features = df.shape[1] - 1
			instances = df.shape[0]
			classified = 0; mae = 0; mse = 0
			
			#instead of for loops, pandas functions perform well
			raw_df['Prediction'] = raw_df.apply(findPrediction, axis=1)
			if algorithm != 'Regression':
				idx = raw_df[raw_df['Prediction'] == raw_df['Decision']].index
				
				#raw_df['Classified'] = 0
				#raw_df.loc[idx, 'Classified'] = 1
				#print(raw_df)
				
				accuracy = 100*len(idx)/instances
				print("Accuracy: ", accuracy,"% on ",instances," instances")
			else:
				raw_df['Absolute_Error'] = abs(raw_df['Prediction'] - raw_df['Decision'])
				raw_df['Absolute_Error_Squared'] = raw_df['Absolute_Error'] * raw_df['Absolute_Error']
				
				#print(raw_df)
				
				mae = raw_df['Absolute_Error'].sum()/instances
				print("MAE: ",mae)
				
				mse = raw_df['Absolute_Error_Squared'].sum()/instances
				rmse = math.sqrt(mse)
				print("RMSE: ",rmse)
				
				mean = raw_df['Decision'].mean()
				print("Mean: ", mean)
				
				if mean > 0:
					print("MAE / Mean: ",100*mae/mean,"%")
					print("RMSE / Mean: ",100*rmse/mean,"%")
	
	return models
			
def findPrediction(row):
	params = []
	num_of_features = row.shape[0] - 1
	for j in range(0, num_of_features):
		params.append(row[j])
	
	moduleName = "outputs/rules/rules"
	fp, pathname, description = imp.find_module(moduleName)
	myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
	
	prediction = myrules.findDecision(params)
	return prediction
	
"""
If you set parelellisim True, then branches will be created parallel. Rules are stored in a json file randomly. This program reconstructs built rules in a tree form. In this way, we can build decision trees faster.
"""

def reconstructRules(source):
	
	#print("Reconstructing ",source)
	
	file_name = source.split(".json")[0]
	file_name = "outputs/rules/"+file_name+".py"
	source = "outputs/rules/"+source
	
	functions.createFile(file_name, "#This rule was reconstructed from "+source+"\n")
	
	with open(source, 'r') as f:
		rules = json.load(f)

	#print(rules)

	def padleft(rule, level):
		for i in range(0, level):
			rule = "\t"+rule
		return rule

	#print("def findDecision(obj):")

	max_level = 0

	rule_set = []
	#json file might not store rules respectively
	for instance in rules:
		if len(instance) > 0:
			rule = []
			rule.append(instance["current_level"])
			rule.append(instance["leaf_id"])
			rule.append(instance["parents"])
			rule.append(instance["rule"])
			rule_set.append(rule)
			#print(padleft(instance["rule"], instance["current_level"]))

	df = np.array(rule_set)
	
	def extractRules(df, parent = 'root', level=1):
	
		level_raw = level * 1; parent_raw = copy.copy(parent)
		
		for i in range(0 ,df.shape[0]):
			leaf_id = df[i][1]
			parent_id = df[i][2]
			rule = df[i][3]
			
			if parent_id == parent:
				functions.storeRule(file_name, padleft(rule, level))
				
				level = level + 1; parent = copy.copy(leaf_id)
				extractRules(df, parent, level)
				level = level_raw * 1; parent = copy.copy(parent_raw) #restore
			
	functions.storeRule(file_name, "def findDecision(obj):")
	extractRules(df)
