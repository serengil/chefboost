import math
import imp
import uuid
import json
import numpy as np
import copy
import multiprocessing
import os
import multiprocessing.pool
import pandas as pd

from chefboost.training import Preprocess
from chefboost.commons import functions, evaluate

#----------------------------------------

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
#----------------------------------------
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
	decision_classes = df["Decision"].unique()
	
	#-----------------------------
	
	if algorithm == 'Regression':
		stdev = df['Decision'].std(ddof=0)
	
	entropy = 0
	
	if algorithm == "ID3" or algorithm == "C4.5":
		entropy = calculateEntropy(df, config)
	#print("entropy: ",entropy)

	columns = df.shape[1]; instances = df.shape[0]

	gains = []; gainratios = []; ginis = []; reducted_stdevs = []; chi_squared_values = []

	for i in range(0, columns-1):
		column_name = df.columns[i]
		column_type = df[column_name].dtypes
		
		#print(column_name,"->",column_type)
		
		if column_type != 'object':
			df = Preprocess.processContinuousFeatures(algorithm, df, column_name, entropy, config)
		
		classes = df[column_name].value_counts()
		
		gain = entropy * 1; splitinfo = 0; gini = 0; weighted_stdev = 0; chi_squared_value = 0
		
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
			
			elif algorithm == 'CHAID':
				num_of_decisions = len(decision_classes)
				
				expected = subset_instances / num_of_decisions
				
				for d in decision_classes:
					num_of_d = subdataset[subdataset["Decision"] == d].shape[0]
					
					chi_square_of_d = math.sqrt(((num_of_d - expected) * (num_of_d - expected)) / expected)
					
					chi_squared_value += chi_square_of_d
				
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
		
		elif algorithm == "CHAID":
			chi_squared_values.append(chi_squared_value)
		
		elif algorithm == 'Regression':
			reducted_stdev = stdev - weighted_stdev
			reducted_stdevs.append(reducted_stdev)
	
	#print(df)
	if algorithm == "ID3":
		winner_index = gains.index(max(gains))
		metric_value = entropy
		metric_name = "Entropy"
	elif algorithm == "C4.5":
		winner_index = gainratios.index(max(gainratios))
		metric_value = entropy
		metric_name = "Entropy"
	elif algorithm == "CART":
		winner_index = ginis.index(min(ginis))
		metric_value = min(ginis)
		metric_name = "Gini"
	elif algorithm == "CHAID":
		winner_index = chi_squared_values.index(max(chi_squared_values))
		metric_value = max(chi_squared_values)
		metric_name = "ChiSquared"
	elif algorithm == "Regression":
		winner_index = reducted_stdevs.index(max(reducted_stdevs))
		metric_value = max(reducted_stdevs)
		metric_name = "Std"
	winner_name = df.columns[winner_index]

	return winner_name, df.shape[0], metric_value, metric_name

def createBranch(config, current_class, subdataset, numericColumn, branch_index
	, winner_name, winner_index, root, parents, file, dataset_features, num_of_instances, metric, tree_id = 0):
	
	algorithm = config['algorithm']
	enableAdaboost = config['enableAdaboost']
	enableGBM = config['enableGBM']
	max_depth = config['max_depth']
	enableParallelism = config['enableParallelism']
	
	charForResp = "'"
	if algorithm == 'Regression':
		charForResp = ""
	
	#---------------------------
	
	json_file = file.split(".")[0]+".json"
	
	tmp_root = root * 1
	parents_raw = copy.copy(parents)
	
	#---------------------------
	
	if numericColumn == True:
		compareTo = current_class #current class might be <=x or >x in this case
	else:
		compareTo = " == '"+str(current_class)+"'"
	
	#print(subdataset)
	
	terminateBuilding = False
	
	#-----------------------------------------------
	#can decision be made?
	
	if enableGBM == True and root >= max_depth: #max depth
		final_decision = subdataset['Decision'].mean()
		terminateBuilding = True
	elif enableAdaboost == True:
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
	
	if enableParallelism == True:
		check_condition = "if" #TODO: elif checks might be above than if statements in parallel
	else:	
		if branch_index == 0:
			check_condition = "if"
		else:
			check_condition = "elif"
	
	check_rule = check_condition+" obj["+str(winner_index)+"]"+compareTo+":"
	
	leaf_id = str(uuid.uuid1())
	custom_rule_file = "outputs/rules/"+str(leaf_id)+".txt"
		
	if enableParallelism != True:
		
		#check_rule += " # feature: "+winner_name+", instances: "+str(num_of_instances)+", "+metric_name+": "+str(round(metric, 4))
		
		functions.storeRule(file,(functions.formatRule(root),"",check_rule))
	else:
		
		sample_rule = {}
		sample_rule["current_level"] = root
		sample_rule["leaf_id"] = leaf_id
		sample_rule["parents"] = parents
		sample_rule["rule"] = check_rule
		sample_rule["feature_idx"] = winner_index
		sample_rule["feature_name"] = winner_name
		sample_rule["instances"] = num_of_instances
		sample_rule["metric"] = metric
		sample_rule["return_statement"] = 0
		sample_rule["tree_id"] = tree_id
		
		#json to string
		sample_rule = json.dumps(sample_rule)
	
		functions.createFile(custom_rule_file, "")
		functions.storeRule(custom_rule_file, sample_rule)
	
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
			sample_rule = {}
			sample_rule["current_level"] = root+1
			sample_rule["leaf_id"] = leaf_id
			sample_rule["parents"] = parents
			sample_rule["rule"] = decision_rule
			sample_rule["feature_idx"] = winner_index
			sample_rule["feature_name"] = winner_name
			sample_rule["instances"] = num_of_instances
			sample_rule["metric"] = 0
			sample_rule["return_statement"] = 1
			sample_rule["tree_id"] = tree_id
			
			#json to string
			sample_rule = ", "+json.dumps(sample_rule)
			
			functions.storeRule(custom_rule_file, sample_rule)
	
	else: #decision is not made, continue to create branch and leafs
		root = root + 1 #the following rule will be included by this rule. increase root
		parents = copy.copy(leaf_id)
		
		buildDecisionTree(subdataset, root, file, config, dataset_features
			, root-1, leaf_id, parents, tree_id = tree_id)
					
		root = tmp_root * 1
		parents = copy.copy(parents_raw)

def buildDecisionTree(df, root, file, config, dataset_features, parent_level = 0, leaf_id = 0, parents = 'root', tree_id = 0, validation_df = None):
	
	models = []
	feature_names = df.columns[0:-1]
	
	enableParallelism = config['enableParallelism']
	algorithm = config['algorithm']
	
	json_file = file.split(".")[0]+".json"
	
	if root == 1:
		if config['enableRandomForest'] != True and config['enableGBM'] != True and config['enableAdaboost'] != True:
			raw_df = df.copy()
	
	#--------------------------------------
	
	df_copy = df.copy()
	
	winner_name, num_of_instances, metric, metric_name = findDecision(df, config)
	
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
	
	num_cores = config["num_cores"]
	
	input_params = []
	
	#serial approach
	for i in range(0,len(classes)):
		current_class = classes[i]
		subdataset = df[df[winner_name] == current_class]
		subdataset = subdataset.drop(columns=[winner_name])
		branch_index = i * 1
		
		#create branches serially
		if enableParallelism != True:
			
			if i == 0:
				#descriptor = "# Feature: "+winner_name+", Instances: "+str(num_of_instances)+", "+metric_name+": "+str(round(metric, 4))
				
				descriptor = {
					"feature": winner_name,
					"instances": num_of_instances,
					#"metric_name": metric_name,
					"metric_value": round(metric, 4),
					"depth": parent_level + 1
				}
				descriptor = "# "+json.dumps(descriptor)
				
				functions.storeRule(file, (functions.formatRule(root), "", descriptor))
			
			createBranch(config, current_class, subdataset, numericColumn, branch_index
				, winner_name, winner_index, root, parents, file, dataset_features, num_of_instances, metric, tree_id = tree_id)
		else:
			input_params.append((config, current_class, subdataset, numericColumn, branch_index
				, winner_name, winner_index, root, parents, file, dataset_features, num_of_instances, metric, tree_id))
	
	#---------------------------
	#add else condition in the decision tree
	
	if df.Decision.dtypes == 'object': #classification
		pivot = pd.DataFrame(subdataset.Decision.value_counts()).reset_index()
		pivot = pivot.rename(columns = {"Decision": "Instances","index": "Decision"})
		pivot = pivot.sort_values(by = ["Instances"], ascending = False).reset_index()
		
		else_decision = "return '%s'" % (pivot.iloc[0].Decision)
		
		if enableParallelism != True:
			functions.storeRule(file,(functions.formatRule(root), "else:"))
			functions.storeRule(file,(functions.formatRule(root+1), else_decision))
		else: #parallelism
			leaf_id = str(uuid.uuid1())
			custom_rule_file = "outputs/rules/"+str(leaf_id)+".txt"
			
			check_rule = "else: "+else_decision
			
			sample_rule = {}
			sample_rule["current_level"] = root
			sample_rule["leaf_id"] = leaf_id
			sample_rule["parents"] = parents
			sample_rule["rule"] = check_rule
			sample_rule["feature_idx"] = -1
			sample_rule["feature_name"] = ""
			sample_rule["instances"] = df.shape[0]
			sample_rule["metric"] = 0
			sample_rule["return_statement"] = 0
			sample_rule["tree_id"] = tree_id
			
			#json to string
			sample_rule = json.dumps(sample_rule)
			
			functions.createFile(custom_rule_file, "")
			functions.storeRule(custom_rule_file, sample_rule)
			
	else: #regression
		else_decision = "return %s" % (subdataset.Decision.mean())
				
		if enableParallelism != True:
			functions.storeRule(file,(functions.formatRule(root), "else:"))
			functions.storeRule(file,(functions.formatRule(root+1), else_decision))
		else:
			leaf_id = str(uuid.uuid1())
			custom_rule_file = "outputs/rules/"+str(leaf_id)+".txt"
			
			check_rule = "else: "+else_decision
			
			sample_rule = {}
			sample_rule["current_level"] = root
			sample_rule["leaf_id"] = leaf_id
			sample_rule["parents"] = parents
			sample_rule["rule"] = check_rule
			sample_rule["tree_id"] = tree_id
			sample_rule["feature_name"] = ""
			sample_rule["instances"] = 0
			sample_rule["metric"] = 0
			sample_rule["return_statement"] = 1
			
			#json to string
			sample_rule = json.dumps(sample_rule)
			
			functions.createFile(custom_rule_file, "")
			functions.storeRule(custom_rule_file, sample_rule)
	
	#---------------------------
	
	#create branches in parallel
	if enableParallelism == True:
		"""
		#this usage causes trouble for recursive functions
		with Pool(number_of_cpus) as pool:
			pool.starmap(createBranch, input_params)
		"""
		
		#calling parallel run recursively causes bottleneck. limit paralel processes.		
		if parent_level == 0: #TODO: this control might be modified based on num of cores.
			
			#print(len(input_params), " branches will be run parallel")
			pool = MyPool(num_cores)
			results = pool.starmap(createBranch, input_params)
			pool.close()
			pool.join()
		else:
			for input_param in input_params:
				createBranch(input_param[0], input_param[1], input_param[2], input_param[3], input_param[4], input_param[5], input_param[6], input_param[7], input_param[8], input_param[9], input_param[10], input_param[11], input_param[12], input_param[13])
	
	#---------------------------------------------
	
	random_forest_enabled = config['enableRandomForest']
	
	if root == 1:
		
		if enableParallelism == True:
			
			#custom rules are stored in .txt files. merge them all in a json file
			
			functions.createFile(json_file, "[\n")
			
			custom_rules = []
						
			file_index = 0
			for file in os.listdir(os.getcwd()+"/outputs/rules"):
				if file.endswith(".txt"):
					
					f = open(os.getcwd()+"/outputs/rules/"+file, "r")
					custom_rule = f.read()
					
					#--------------------------------
					if random_forest_enabled:
						#read tree_id in the json object
						try:
							custom_rule_json = json.loads(custom_rule)
						except:
							#return statements include two json objects as string
							custom_rule_json = json.loads("["+custom_rule+"]")
							
						try:
							source_tree_id = custom_rule_json["tree_id"]
						except:	
							source_tree_id = custom_rule_json[0]["tree_id"]
						
					#--------------------------------
					
					if random_forest_enabled != True or (random_forest_enabled == True and source_tree_id == tree_id):
						custom_rules.append(os.getcwd()+"/outputs/rules/"+file)
						
						if file_index > 0:
							custom_rule = ", "+custom_rule
							
						functions.storeRule(json_file, custom_rule)
						file_index = file_index + 1
					
					f.close()
					
			functions.storeRule(json_file, "]")
			
			#-----------------------------------
			
			#custom rules are already merged in a json file. clear messy custom rules
			#TO-DO: if random forest trees are handled in parallel, this would be a problem. You cannot know the related tree of a rule. You should store a global tree id in a rule.
			
			
			for file in custom_rules:
				os.remove(file)
			
			#-----------------------------------
			
			reconstructRules(json_file, feature_names)
			
			#-----------------------------------
		
		#is regular decision tree
		if config['enableRandomForest'] != True and config['enableGBM'] != True and config['enableAdaboost'] != True:
		#this is reguler decision tree. find accuracy here.
			
			moduleName = "outputs/rules/rules"
			fp, pathname, description = imp.find_module(moduleName)
			myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
			models.append(myrules)
			
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

def reconstructRules(source, feature_names, tree_id = 0):
	
	#print("Reconstructing ",source)
	
	file_name = source.split(".json")[0]
	file_name = file_name+".py"
	
	#-----------------------------------
	
	constructor = "def findDecision(obj): #"
	idx = 0
	for feature in feature_names:
		constructor = constructor + "obj["+str(idx)+"]: "+feature
		
		if idx < len(feature_names) - 1:
			constructor = constructor+", "
		idx = idx + 1
	
	functions.createFile(file_name, constructor+"\n")
	
	#-----------------------------------
	
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
			rule.append(instance["feature_name"])
			rule.append(instance["instances"])
			rule.append(instance["metric"])
			rule.append(instance["return_statement"])
			rule_set.append(rule)
			#print(padleft(instance["rule"], instance["current_level"]))

	df = np.array(rule_set)
	
	def extractRules(df, parent = 'root', level=1):
	
		level_raw = level * 1; parent_raw = copy.copy(parent)
		
		else_rule = ""
		
		leaf_idx = 0
		for i in range(0 ,df.shape[0]):
			current_level = int(df[i][0])
			leaf_id = df[i][1]
			parent_id = df[i][2]
			rule = df[i][3]
			feature_name = df[i][4]
			instances = int(df[i][5])
			metric = float(df[i][6])
			return_statement = int(df[i][7])
			
			if parent_id == parent:
				
				if_statement = False
				if rule[0:2] == "if":
					if_statement = True
				
				else_statement = False
				if rule[0:5] == "else:":
					else_statement = True
					else_rule = rule
				
				#------------------------
				
				if else_statement != True:
				
					if if_statement == True and leaf_idx > 0:
						rule = "el"+rule
					
					#print(padleft(rule, level), "(", leaf_idx,")")
					
					if leaf_idx == 0 and return_statement == 0:
						explainer = {}
						explainer["feature"] = feature_name
						explainer["instances"] = instances
						explainer["metric_value"] = round(metric, 4)
						explainer["depth"] = current_level
						explainer = "# "+json.dumps(explainer)
						functions.storeRule(file_name, padleft(explainer, level))
					
					functions.storeRule(file_name, padleft(rule, level))
					
					level = level + 1; parent = copy.copy(leaf_id)
					extractRules(df, parent, level)
					level = level_raw * 1; parent = copy.copy(parent_raw) #restore
					
					leaf_idx = leaf_idx + 1
		
		#add else statement
		
		if else_rule != "":
			#print(padleft(else_rule, level))
			functions.storeRule(file_name, padleft(else_rule, level))
			
	#------------------------------------
	
	extractRules(df)

	#------------------------------------
