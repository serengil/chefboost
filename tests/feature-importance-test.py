#!pip install chefboost
from chefboost import Chefboost as cb
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
	df = pd.read_csv("dataset/golf.txt")
	config = config = {'algorithm': 'C4.5', 'enableParallelism': True}

	model = cb.fit(df, config)
	
	fi = cb.feature_importance()
	print(fi)