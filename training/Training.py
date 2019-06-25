import math
import imp

from training import Preprocess
from commons import functions

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

def buildDecisionTree(df,root,file, config, dataset_features):

	models = []

	if root == 1:
		if config['enableRandomForest'] != True and config['enableGBM'] != True and config['enableAdaboost'] != True:
			raw_df = df.copy()
	
	algorithm = config['algorithm']
	enableAdaboost = config['enableAdaboost']
	
	#--------------------------------------
	
	#print(df.shape)
	charForResp = "'"
	if algorithm == 'Regression':
		charForResp = ""

	tmp_root = root * 1
	
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
		
		functions.storeRule(file,(functions.formatRule(root),"",check_condition," obj[",str(winner_index),"]",compareTo,":"))
		
		#-----------------------------------------------
		
		if terminateBuilding == True: #check decision is made
			functions.storeRule(file,(functions.formatRule(root+1),"return ",charForResp+str(final_decision)+charForResp))
			
		else: #decision is not made, continue to create branch and leafs
			root = root + 1 #the following rule will be included by this rule. increase root
			buildDecisionTree(subdataset, root, file, config, dataset_features)
		
		root = tmp_root * 1
	
	#---------------------------------------------
	
	#calculate accuracy metrics
	if root == 1:
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
	
