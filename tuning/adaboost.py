import pandas as pd
import numpy as np

from commons import functions
from training import Training

def apply(df, config, header, dataset_features):
	
	debug = config['debug']
	
	#------------------------
	
	rows = df.shape[0]; columns = df.shape[1]
	final_predictions = pd.DataFrame(np.zeros([rows, 1]), columns=['prediction'])
	
	worksheet = df.copy()
	worksheet['weight'] = 1 #/ rows
	
	tmp_df = df.copy()
	tmp_df['Decision'] = worksheet['weight'] * tmp_df['Decision'] #normal distribution
	
	for i in range(0, 1):	
		root = 1
		file = "outputs/rules/rules_"+str(i)+".py"
		
		if debug == False: functions.createFile(file, header)
		
		#print(tmp_df)
		Training.buildDecisionTree(tmp_df, root, file, config, dataset_features)
	
	#print(final_predictions)
	
	"""for row, instance in final_predictions.iterrows():
		print("actual: ",df.loc[row]['Decision'],", prediction: ",functions.sign(instance['prediction'])," (",df.loc[row]['Decision'] == functions.sign(instance['prediction']),")")"""