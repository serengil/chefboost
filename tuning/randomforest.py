import pandas as pd

from multiprocessing import Pool

from commons import functions
from training import Training
	
def apply(df, config, header, dataset_features):
	
	debug = config['debug'] 
	num_of_trees = config['num_of_trees']
	
	for i in range(0, num_of_trees):
		subset = df.sample(frac=1/num_of_trees)
		
		root = 1
		
		file = "outputs/rules/rule_"+str(i)+".py"
		
		if debug == False:
			functions.createFile(file, header)
		
		Training.buildDecisionTree(subset,root, file, config, dataset_features)
