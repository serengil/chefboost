import pandas as pd
import numpy as np

from chefboost.commons import functions, evaluate
from chefboost.training import Training
from chefboost import Chefboost as cb

import imp
import math

from tqdm import tqdm

def findPrediction(row):
	epoch = row['Epoch']
	row = row.drop(labels=['Epoch'])
	columns = row.shape[0]
	
	params = []
	for j in range(0, columns-1):
		params.append(row[j])
		
	moduleName = "outputs/rules/rules_%d" % (epoch)
	fp, pathname, description = imp.find_module(moduleName)
	myrules = imp.load_module(moduleName, fp, pathname, description)
	
	prediction = functions.sign(myrules.findDecision(params))
	
	return prediction

def apply(df, config, header, dataset_features, validation_df = None, process_id = None):

	models = []; alphas = []
	
	initializeAlphaFile()
	
	num_of_weak_classifier = config['num_of_weak_classifier']
	
	#------------------------
	
	rows = df.shape[0]; columns = df.shape[1]
	final_predictions = pd.DataFrame(np.zeros([rows, 1]), columns=['prediction'])
	
	worksheet = df.copy()
	worksheet['Weight'] = 1 / rows #uniform distribution initially
	
	final_predictions = pd.DataFrame(np.zeros((df.shape[0], 2)), columns = ['Prediction', 'Actual'])
	final_predictions['Actual'] = df['Decision']
	
	best_epoch_idx = 0; best_epoch_value = 1000000
	
	#for i in range(0, num_of_weak_classifier):
	pbar = tqdm(range(0, num_of_weak_classifier), desc='Adaboosting')
	for i in pbar:
		worksheet['Decision'] = worksheet['Weight'] * worksheet['Decision']
		
		root = 1
		file = "outputs/rules/rules_"+str(i)+".py"
		
		functions.createFile(file, header)
		
		#print(worksheet)
		Training.buildDecisionTree(worksheet.drop(columns=['Weight'])
			, root, file, config, dataset_features
			, parent_level = 0, leaf_id = 0, parents = 'root', main_process_id = process_id)
		
		#---------------------------------------
		
		moduleName = "outputs/rules/rules_"+str(i)
		fp, pathname, description = imp.find_module(moduleName)
		myrules = imp.load_module(moduleName, fp, pathname, description)
		models.append(myrules)
		
		#---------------------------------------
		
		df['Epoch'] = i
		worksheet['Prediction'] = df.apply(findPrediction, axis=1)
		df = df.drop(columns = ['Epoch'])
		
		#---------------------------------------
		worksheet['Actual'] = df['Decision']
		worksheet['Loss'] = abs(worksheet['Actual'] - worksheet['Prediction'])/2
		worksheet['Weight_Times_Loss'] = worksheet['Loss'] * worksheet['Weight']
		
		epsilon = worksheet['Weight_Times_Loss'].sum()
		alpha = math.log((1 - epsilon)/epsilon)/2 #use alpha to update weights in the next round
		alphas.append(alpha)
		
		#-----------------------------
		
		#store alpha
		addEpochAlpha(i, alpha)
		
		#-----------------------------
		
		worksheet['Alpha'] = alpha
		worksheet['New_Weights'] = worksheet['Weight'] * (-alpha * worksheet['Actual'] * worksheet['Prediction']).apply(math.exp)
		
		#normalize
		worksheet['New_Weights'] = worksheet['New_Weights'] / worksheet['New_Weights'].sum()
		worksheet['Weight'] = worksheet['New_Weights']
		worksheet['Decision'] = df['Decision']
		
		final_predictions['Prediction']  =  final_predictions['Prediction'] + worksheet['Alpha'] * worksheet['Prediction']
		#print(final_predictions)
		worksheet = worksheet.drop(columns = ['New_Weights', 'Prediction', 'Actual', 'Loss', 'Weight_Times_Loss', 'Alpha'])
		
		mae = (np.abs(final_predictions['Prediction'].apply(functions.sign) - final_predictions['Actual'])/2).sum()/final_predictions.shape[0]
		#print(mae)
		
		if mae < best_epoch_value:
			best_epoch_value = mae * 1
			best_epoch_idx = i * 1
		
		pbar.set_description("Epoch %d. Loss: %d. Process: " % (i+1, mae))
	
	#------------------------------
	
	print("The best epoch is ",best_epoch_idx," with the ",best_epoch_value," MAE score")
	
	models = models[0: best_epoch_idx+1]
	alphas = alphas[0: best_epoch_idx+1]
	
	#------------------------------
	
	return models, alphas

def initializeAlphaFile():
	file = "outputs/rules/alphas.py"
	header = "def findAlpha(epoch):\n"
	functions.createFile(file, header)

def addEpochAlpha(epoch, alpha):
	file = "outputs/rules/alphas.py"
	content = "   if epoch == "+str(epoch)+":\n"
	content += "      return "+str(alpha)
	functions.storeRule(file, content)
