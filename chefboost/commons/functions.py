import numpy as np
import pathlib
import imp
import os
from os import path

def restoreTree(moduleName):
   fp, pathname, description = imp.find_module(moduleName)
   return imp.load_module(moduleName, fp, pathname, description)

def softmax(w):
	e = np.exp(np.array(w, dtype=np.float32))
	dist = e / np.sum(e)
	return dist

def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

def formatRule(root):
	resp = ''
	
	for i in range(0, root):
		resp = resp + '   '
	
	return resp	

def storeRule(file,content):
	f = open(file, "a+")
	f.writelines(content)
	f.writelines("\n")

def createFile(file,content):
	f = open(file, "w")
	f.write(content)

def initializeFolders():
	import sys
	sys.path.append("..")
	pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
	pathlib.Path("outputs/data").mkdir(parents=True, exist_ok=True)
	pathlib.Path("outputs/rules").mkdir(parents=True, exist_ok=True)
	
	#-----------------------------------
	
	#clear existing rules in outputs/
		
	outputs_path = os.getcwd()+os.path.sep+"outputs"+os.path.sep
	
	try:
		if path.exists(outputs_path+"data"):
			for file in os.listdir(outputs_path+"data"):
				os.remove(outputs_path+"data"+os.path.sep+file)
		
		if path.exists(outputs_path+"rules"):
			for file in os.listdir(outputs_path+"rules"):
				if ".py" in file or ".json" in file or ".txt" in file or ".pkl" in file:
					os.remove(outputs_path+"rules"+os.path.sep+file)
	except Exception as err:
		print("WARNING: ", str(err))
	
	#------------------------------------
	

def initializeParams(config):
	algorithm = 'ID3'
	enableRandomForest = False; num_of_trees = 5; enableMultitasking = False
	enableGBM = False; epochs = 10; learning_rate = 1
	enableAdaboost = False; num_of_weak_classifier = 4
	enableParallelism = True
	
	for key, value in config.items():
		if key == 'algorithm':
			algorithm = value
		#---------------------------------	
		elif key == 'enableRandomForest':
			enableRandomForest = value
		elif key == 'num_of_trees':
			num_of_trees = value
		elif key == 'enableMultitasking':
			enableMultitasking = value
		#---------------------------------
		elif key == 'enableGBM':
			enableGBM = value
		elif key == 'epochs':
			epochs = value
		elif key == 'learning_rate':
			learning_rate = value
		#---------------------------------	
		elif key == 'enableAdaboost':
			enableAdaboost = value
		elif key == 'num_of_weak_classifier':
			num_of_weak_classifier = value
		#---------------------------------	
		elif key == 'enableParallelism':
			enableParallelism = value
			
	config['algorithm'] = algorithm
	config['enableRandomForest'] = enableRandomForest
	config['num_of_trees'] = num_of_trees
	config['enableMultitasking'] = enableMultitasking
	config['enableGBM'] = enableGBM
	config['epochs'] = epochs
	config['learning_rate'] = learning_rate
	config['enableAdaboost'] = enableAdaboost
	config['num_of_weak_classifier'] = num_of_weak_classifier
	config['enableParallelism'] = enableParallelism
	
	return config
