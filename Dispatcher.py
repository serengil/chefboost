import Chefboost as chef
import pandas as pd

mae = 0

#------------------------------------

#gradient boosting regression tree
config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1}

df = pd.read_csv("dataset/golf4.txt")
models = chef.fit(df.copy(), config)

for index, instance in df.iterrows():
	actual = instance['Decision']
	prediction = chef.predict(models, instance.values)
	
	error = abs(actual - prediction)
	mae += error
	print("Prediction: ",prediction,", Actual: ",actual)

print("MAE: ",mae/df.shape[0])

#------------------------------------

"""
config = {'algorithm': 'C4.5'}
model = chef.fit(pd.read_csv("dataset/golf2.txt"), config)

prediction = chef.predict(model, ['Sunny',85,85,'Weak'])
print(prediction)
"""

#------------------------------------
"""
config = {'algorithm': 'C4.5'}
model = chef.fit(pd.read_csv("dataset/golf3.txt"), config)

prediction = chef.predict(model, ['Sunny', 'Hot', 'High', 'Weak'])
print(prediction)
"""

#------------------------------------

"""
config = {'algorithm': 'ID3'}
model = chef.fit(pd.read_csv("dataset/golf.txt"), config)
prediction = chef.predict(model, ['Sunny', 'Hot', 'High', 'Weak'])
print(prediction)
"""

#---------------------------------
"""
print("Adaboost")
config = {'enableAdaboost': True, 'num_of_weak_classifier': 4}
models, alphas = chef.fit(pd.read_csv("dataset/adaboost.txt"), config)
prediction = chef.predict(models, [4, 3.5], alphas)
print(prediction)
"""

#------------------------------------

"""
#random forest - classification
config = {'algorithm': 'ID3', 'enableRandomForest': True, 'num_of_trees': 5,}
models = chef.fit(pd.read_csv("dataset/car.data"), config)

prediction = chef.predict(models, ['vhigh','vhigh',2,'2','small','low'])
print(prediction)

prediction = chef.predict(models, ['high','high',4,'more','big','high'])
print(prediction)
"""

#------------------------------------
"""
#classification
config = {'algorithm': 'ID3', 'enableGBM': True, 'epochs': 5, 'learning_rate': 1}

chef.fit(pd.read_csv("dataset/iris.data", names=["Sepal length","Sepal width","Petal length","Petal width","Decision"]), config)
"""