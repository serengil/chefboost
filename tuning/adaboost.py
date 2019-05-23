import pandas as pd
import numpy as np

from commons import functions
from training import Training

import imp
import math

from tqdm import tqdm

def apply(df, config, header, dataset_features):
	
	debug = config['debug']
	num_of_weak_classifier = config['num_of_weak_classifier']
	
	#------------------------
	
	rows = df.shape[0]; columns = df.shape[1]
	final_predictions = pd.DataFrame(np.zeros([rows, 1]), columns=['prediction'])
	
	worksheet = df.copy()
	worksheet['Weight'] = 1 / rows #uniform distribution initially
	
	final_predictions = pd.DataFrame(np.zeros((df.shape[0], 2)), columns = ['Prediction', 'Actual'])
	final_predictions['Actual'] = df['Decision']
	
	#for i in range(0, num_of_weak_classifier):
	pbar = tqdm(range(0, num_of_weak_classifier), desc='Adaboosting')
	for i in pbar:
		worksheet['Decision'] = worksheet['Weight'] * worksheet['Decision']
		
		root = 1
		file = "outputs/rules/rules_"+str(i)+".py"
		
		if debug == False: functions.createFile(file, header)
		
		#print(worksheet)
		Training.buildDecisionTree(worksheet.drop(columns=['Weight'])
			, root, file, config, dataset_features)
			
		moduleName = "outputs/rules/rules_"+str(i)
		fp, pathname, description = imp.find_module(moduleName)
		myrules = imp.load_module(moduleName, fp, pathname, description)
		
		predictions = []
		for index, instance in df.iterrows():
			params = []
			for j in range(0, columns-1):
				params.append(instance[j])
		
			prediction = functions.sign(myrules.findDecision(params))
			predictions.append(prediction)
		
		worksheet['Prediction'] = pd.Series(predictions)
		worksheet['Actual'] = df['Decision']
		worksheet['Loss'] = abs(worksheet['Actual'] - worksheet['Prediction'])/2
		worksheet['Weight_Times_Loss'] = worksheet['Loss'] * worksheet['Weight']
		
		epsilon = worksheet['Weight_Times_Loss'].sum()
		alpha = math.log((1 - epsilon)/epsilon)/2 #use alpha to update weights in the next round
		
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
		pbar.set_description("Epoch %d. Loss: %d. Process: " % (i+1, mae))
	
	#------------------------------
	final_predictions['Prediction'] = final_predictions['Prediction'].apply(functions.sign)
	final_predictions['Absolute_Error'] = np.abs(final_predictions['Actual'] - final_predictions['Prediction'])/2
	print(final_predictions)
	mae = final_predictions['Absolute_Error'].sum() / final_predictions.shape[0]
	print("Loss (MAE) found ", mae, " with ",num_of_weak_classifier, ' weak classifiers')
