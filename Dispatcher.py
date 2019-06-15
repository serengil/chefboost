import Chefboost as chef
import pandas as pd
import numpy as np

mae = 0

#------------------------------------

#gradient boosting regression tree
config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1}

df = pd.read_csv("dataset/golf4.txt")
models = chef.fit(df.copy(), config)

for index, instance in df.iterrows():
	actual = instance['Decision']
	prediction = chef.predict(config, models, instance.values)
	
	error = abs(actual - prediction)
	mae += error
	print("Prediction: ",prediction,", Actual: ",actual)

print("MAE: ",mae/df.shape[0])

#------------------------------------

#gradient boosting classification
config = {'algorithm': 'ID3', 'enableGBM': True, 'epochs': 5, 'learning_rate': 1}

models, classes = chef.fit(pd.read_csv("dataset/iris.data", names=["Sepal length","Sepal width","Petal length","Petal width","Decision"]), config)

test_set = [7.0,3.2,4.7,1.4]

prediction = chef.predict(config, models, test_set, classes)
print(prediction)

#------------------------------------

#regular decision tree
config = {'algorithm': 'C4.5'}
model = chef.fit(pd.read_csv("dataset/golf2.txt"), config)

prediction = chef.predict(config, model, ['Sunny',85,85,'Weak'])
print(prediction)

#------------------------------------

#regular decision tree
config = {'algorithm': 'ID3'}
model = chef.fit(pd.read_csv("dataset/golf.txt"), config)
prediction = chef.predict(config, model, ['Sunny', 'Hot', 'High', 'Weak'])
print(prediction)

#------------------------------------

#regular regression tree
config = {'algorithm': 'Regression'}
model = chef.fit(pd.read_csv("dataset/golf3.txt"), config)

prediction = chef.predict(config, model, ['Sunny', 'Hot', 'High', 'Weak'])
print(prediction)

#---------------------------------

#adaboost
config = {'enableAdaboost': True, 'num_of_weak_classifier': 4}
models, alphas = chef.fit(pd.read_csv("dataset/adaboost.txt"), config)
prediction = chef.predict(config, models, [4, 3.5], alphas)
print(prediction)

#------------------------------------

#random forest - classification
config = {'algorithm': 'ID3', 'enableRandomForest': True, 'num_of_trees': 5,}
models = chef.fit(pd.read_csv("dataset/car.data"), config)

prediction = chef.predict(config, models, ['vhigh','vhigh',2,'2','small','low'])
print(prediction)

prediction = chef.predict(config, models, ['high','high',4,'more','big','high'])
print(prediction)

#------------------------------------