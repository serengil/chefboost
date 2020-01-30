import pandas as pd
import numpy as np

import imp

from chefboost.commons import functions
from chefboost.training import Preprocess, Training

from tqdm import tqdm

def findPrediction(row):
	epoch = row['Epoch']
	row = row.drop(labels=['Epoch'])
	columns = row.shape[0]
	
	params = []
	for j in range(0, columns-1):
		params.append(row[j])
		
	moduleName = "outputs/rules/rules%s" % (epoch-1)
	fp, pathname, description = imp.find_module(moduleName)
	myrules = imp.load_module(moduleName, fp, pathname, description)
	
	prediction = int(myrules.findDecision(params)) 
	
	return prediction

def regressor(df, config, header, dataset_features):
	models = []
	
	algorithm = config['algorithm']
	
	enableRandomForest = config['enableRandomForest']
	num_of_trees = config['num_of_trees']
	enableMultitasking = config['enableMultitasking']

	enableGBM = config['enableGBM']
	epochs = config['epochs']
	learning_rate = config['learning_rate']

	enableAdaboost = config['enableAdaboost']
	
	#------------------------------
	
	boosted_from = 0; boosted_to = 0
	
	#------------------------------
	
	base_df = df.copy()
	
	#gbm will manipulate actuals. store its raw version.
	target_values = base_df['Decision'].values
	num_of_instances = target_values.shape[0]
	
	root = 1
	file = "outputs/rules/rules0.py"; json_file = "outputs/rules/rules0.json"
	functions.createFile(file, header)
	functions.createFile(json_file, "[\n")
	
	Training.buildDecisionTree(df,root,file, config, dataset_features
		, parent_level = 0, leaf_id = 0, parents = 'root') #generate rules0
	
	#functions.storeRule(json_file," {}]")
	
	df = base_df.copy()
	
	base_df['Boosted_Prediction'] = 0
	
	#------------------------------
	
	pbar = tqdm(range(1,epochs+1), desc='Boosting')
	
	#for index in range(1,epochs+1):
	#for index in tqdm(range(1,epochs+1), desc='Boosting'):
	for index in pbar:
		#print("epoch ",index," - ",end='')
		loss = 0
		
		#run data(i-1) and rules(i-1), save data1
		
		#dynamic import
		moduleName = "outputs/rules/rules%s" % (index-1)
		fp, pathname, description = imp.find_module(moduleName)
		myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
		
		models.append(myrules)
		
		new_data_set = "outputs/data/data%s.csv" % (index)
		f = open(new_data_set, "w")
		
		#put header in the following file
		columns = df.shape[1]
		
		mae = 0
		
		#----------------------------------------
		
		df['Epoch'] = index
		df['Prediction'] = df.apply(findPrediction, axis=1)
		
		base_df['Boosted_Prediction'] += df['Prediction']
		
		loss = (base_df['Boosted_Prediction'] - base_df['Decision']).pow(2).sum()
		
		if index == 1: 
			boosted_from = loss / num_of_instances
		elif index == epochs:
			boosted_to = loss / num_of_instances
		
		df['Decision'] = int(learning_rate)*(df['Decision'] - df['Prediction'])
		df = df.drop(columns = ['Epoch', 'Prediction'])
		
		#---------------------------------
		
		df.to_csv(new_data_set, index=False)
		#data(i) created
		
		#---------------------------------
		
		file = "outputs/rules/rules"+str(index)+".py"
		json_file = "outputs/rules/rules"+str(index)+".json"
		
		functions.createFile(file, header)
		functions.createFile(json_file, "[\n")
		
		current_df = df.copy()
		Training.buildDecisionTree(df,root,file, config, dataset_features
			, parent_level = 0, leaf_id = 0, parents = 'root')
		
		#functions.storeRule(json_file," {}]")
		
		df = current_df.copy() #numeric features require this restoration to apply findDecision function
		
		#rules(i) created
		
		loss = loss / num_of_instances
		#print("epoch ",index," - loss: ",loss)
		#print("loss: ",loss)
		pbar.set_description("Epoch %d. Loss: %d. Process: " % (index, loss))
		
		#---------------------------------
	
	print("MSE of ",num_of_instances," instances are boosted from ",boosted_from," to ",boosted_to," in ",epochs," epochs")
	
	return models

def classifier(df, config, header, dataset_features):
	
	models = []
	
	print("gradient boosting for classification")
	
	epochs = config['epochs']
	enableParallelism = config['enableParallelism']
	
	temp_df = df.copy()
	original_dataset = df.copy()
	worksheet = df.copy()
	
	classes = df['Decision'].unique()
	
	boosted_predictions = np.zeros([df.shape[0], len(classes)])
	
	pbar = tqdm(range(0, epochs), desc='Boosting')
	
	#store actual set, we will use this to calculate loss
	actual_set = pd.DataFrame(np.zeros([df.shape[0], len(classes)]), columns=classes)
	for i in range(0, len(classes)):
		current_class = classes[i]
		actual_set[current_class] = np.where(df['Decision'] == current_class, 1, 0)
	actual_set = actual_set.values #transform it to numpy array
	
	#for epoch in range(0, epochs):
	for epoch in pbar:
		for i in range(0, len(classes)):
			current_class = classes[i]
			
			if epoch == 0:
				temp_df['Decision'] = np.where(df['Decision'] == current_class, 1, 0)
				worksheet['Y_'+str(i)] = temp_df['Decision']
			else:
				temp_df['Decision'] = worksheet['Y-P_'+str(i)]
			
			predictions = []
			
			#change data type for decision column
			temp_df[['Decision']].astype('int64')
			
			root = 1
			file_base = "outputs/rules/rules-for-"+current_class+"-round-"+str(epoch)
			
			file = file_base+".py"
			functions.createFile(file, header)
			
			if enableParallelism == True:
				json_file = file_base+".json"
				functions.createFile(json_file, "[\n")
			
			Training.buildDecisionTree(temp_df, root, file, config, dataset_features
				, parent_level = 0, leaf_id = 0, parents = 'root')
				
			#decision rules created
			#----------------------------
			
			#dynamic import
			moduleName = "outputs/rules/rules-for-"+current_class+"-round-"+str(epoch)
			fp, pathname, description = imp.find_module(moduleName)
			myrules = imp.load_module(moduleName, fp, pathname, description) #rules0
			
			models.append(myrules)
			
			num_of_columns = df.shape[1]
			
			for row, instance in df.iterrows():
				features = []
				for j in range(0, num_of_columns-1): #iterate on features
					features.append(instance[j])
				
				actual = temp_df.loc[row]['Decision']
				prediction = myrules.findDecision(features)
								
				predictions.append(prediction)
					
			#----------------------------
			if epoch == 0:
				worksheet['F_'+str(i)] = 0
			else:
				worksheet['F_'+str(i)] = pd.Series(predictions).values
			
			boosted_predictions[:,i] = boosted_predictions[:,i] + worksheet['F_'+str(i)].values.astype(np.float32)
			
			#print(boosted_predictions[0:5,:])
			
			worksheet['P_'+str(i)] = 0
			
			#----------------------------
			temp_df = df.copy() #restoration
		
		for row, instance in worksheet.iterrows():
			f_scores = []
			for i in range(0, len(classes)):
				f_scores.append(instance['F_'+str(i)])
							
			probabilities = functions.softmax(f_scores)
							
			for j in range(0, len(probabilities)):
				instance['P_'+str(j)] = probabilities[j]
			
			worksheet.loc[row] = instance
		
		for i in range(0, len(classes)):
			worksheet['Y-P_'+str(i)] = worksheet['Y_'+str(i)] - worksheet['P_'+str(i)]
		
		prediction_set = np.zeros([df.shape[0], len(classes)])
		for i in range(0, boosted_predictions.shape[0]):
			predicted_index = np.argmax(boosted_predictions[i])
			prediction_set[i][predicted_index] = 1
		
		#----------------------------
		#find loss for this epoch: prediction_set vs actual_set
		classified = 0
		for i in range(0, actual_set.shape[0]):
			actual = np.argmax(actual_set[i])
			prediction = np.argmax(prediction_set[i])
			#print("actual: ",actual," - prediction: ",prediction)
			
			if actual == prediction:
				classified = classified + 1
		
		accuracy = str(100 * classified / actual_set.shape[0]) + "%"
		
		#----------------------------
		
		#print(worksheet.head())
		#print("round ",epoch+1)
		pbar.set_description("Epoch %d. Accuracy: %s. Process: " % (epoch+1, accuracy))
	
	return models, classes
