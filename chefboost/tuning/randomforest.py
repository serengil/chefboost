import pandas as pd
import numpy as np
from multiprocessing import Pool
import multiprocessing
from contextlib import closing
from chefboost.commons import functions, evaluate
from chefboost.training import Training
from chefboost import Chefboost as cb
from tqdm import tqdm
import imp
import os

def apply(df, config, header, dataset_features, validation_df = None, process_id = None):

	models = []

	num_of_trees = config['num_of_trees']

	parallelism_on = config["enableParallelism"]

	#TODO: is this logical for 48x2 cores?
	#config["enableParallelism"] = False #run each tree in parallel but each branch in serial

	#TODO: reconstruct for parallel run is problematic. you should reconstruct based on tree id.

	input_params = []

	pbar = tqdm(range(0, num_of_trees), desc='Bagging')
	for i in pbar:
		pbar.set_description("Sub decision tree %d is processing" % (i+1))
		subset = df.sample(frac=1/num_of_trees)

		root = 1

		moduleName = "outputs/rules/rule_"+str(i)
		file = moduleName+".py"

		functions.createFile(file, header)

		if parallelism_on: #parallel run
			input_params.append((subset, root, file, config, dataset_features, 0, 0, 'root', i, None, process_id))

		else: #serial run
			Training.buildDecisionTree(subset,root, file, config, dataset_features, parent_level = 0, leaf_id = 0, parents = 'root', tree_id = i, main_process_id = process_id)

	#-------------------------------

	if parallelism_on:
		num_cores = config["num_cores"]

		#---------------------------------

		if num_of_trees <= num_cores:
			POOL_SIZE = num_of_trees
		else:
			POOL_SIZE = num_cores

		with closing(multiprocessing.Pool(POOL_SIZE)) as pool:
			funclist = []
			for input_param in input_params:
				f = pool.apply_async(buildDecisionTree, [*input_param])
				funclist.append(f)

			#all functions registered here
			#results = []
			for f in tqdm(funclist):
				branch_results = f.get(timeout = 100000)
				#results.append(branch_results)

			pool.close()
			pool.terminate()

	#-------------------------------
	#collect models for both serial and parallel here
	for i in range(0, num_of_trees):
		moduleName = "outputs/rules/rule_"+str(i)
		fp, pathname, description = imp.find_module(moduleName)
		myrules = imp.load_module(moduleName, fp, pathname, description)
		models.append(myrules)

	#-------------------------------

	return models

#wrapper for parallel run
def buildDecisionTree(df, root, file, config, dataset_features, parent_level, leaf_id, parents, tree_id, validation_df = None, process_id = None):
	Training.buildDecisionTree(df, root, file, config, dataset_features, parent_level = parent_level, leaf_id =leaf_id, parents = parents, tree_id = tree_id, main_process_id = process_id)

"""
def buildDecisionTreeWrapper(func, args):
	return func(*args)
"""
