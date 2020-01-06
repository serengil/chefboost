import Chefboost as chef
import pandas as pd
import numpy as np

#------------------------------------

#gradient boosting regression tree
config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1}

df = pd.read_csv("dataset/golf4.txt")
model = chef.fit(df.copy(), config)

chef.save_model(model)

prediction = chef.predict(model, ['Sunny',85,85,'Weak'])
print("Prediction: ",prediction)

restored_model = chef.load_model("model.pkl")
prediction = chef.predict(restored_model, ['Sunny',85,85,'Weak'])
print("Prediction: ",prediction)

#------------------------------------

#gradient boosting classification
config = {'algorithm': 'ID3', 'enableGBM': True, 'epochs': 5, 'learning_rate': 1}

model = chef.fit(pd.read_csv("dataset/iris.data", names=["Sepal length","Sepal width","Petal length","Petal width","Decision"]), config)

test_set = [7.0,3.2,4.7,1.4]

prediction = chef.predict(model, test_set)
print(prediction)

#------------------------------------

#regular decision tree
config = {'algorithm': 'C4.5'}
model = chef.fit(pd.read_csv("dataset/golf2.txt"), config)

prediction = chef.predict(model, ['Sunny',85,85,'Weak'])
#prediction = model[0].findDecision(['Sunny',85,85,'Weak'])
print(prediction)

#------------------------------------

#regular decision tree
config = {'algorithm': 'ID3'}
model = chef.fit(pd.read_csv("dataset/golf2.txt"), config)

prediction = chef.predict(model, ['Sunny',85,85,'Weak'])
#prediction = model[0].findDecision(['Sunny',85,85,'Weak'])
print(prediction)

#------------------------------------

#regular decision tree
config = {'algorithm': 'CART'}
model = chef.fit(pd.read_csv("dataset/golf.txt"), config)

prediction = chef.predict(model, ['Sunny','Hot','High','Weak'])
print(prediction)

#------------------------------------

#regular decision tree
config = {'algorithm': 'ID3'}
model = chef.fit(pd.read_csv("dataset/golf.txt"), config)
prediction = chef.predict(model, ['Sunny', 'Hot', 'High', 'Weak'])
print(prediction)

#------------------------------------

#regular regression tree
config = {'algorithm': 'Regression'}
model = chef.fit(pd.read_csv("dataset/golf3.txt"), config)

prediction = chef.predict(model, ['Sunny', 'Hot', 'High', 'Weak'])
print(prediction)

#---------------------------------

#adaboost
config = {'enableAdaboost': True, 'num_of_weak_classifier': 4}
model = chef.fit(pd.read_csv("dataset/adaboost.txt"), config)
prediction = chef.predict(model, [4, 3.5])
print(prediction)

#------------------------------------

#random forest - classification
config = {'algorithm': 'ID3', 'enableRandomForest': True, 'num_of_trees': 5,}
model = chef.fit(pd.read_csv("dataset/car.data"), config)

prediction = chef.predict(model, ['vhigh','vhigh',2,'2','small','low'])
print(prediction)

prediction = chef.predict(model, ['high','high',4,'more','big','high'])
print(prediction)

#------------------------------------
