import pandas as pd

import sys
sys.path.append("..")

import Chefboost as cb

print("ID3 for nominal features and target:")
config = {'algorithm': 'ID3'}
cb.fit(pd.read_csv("../dataset/golf.txt"), config)

print("-------------------------")

print("ID3 for nominal/numeric features and target:")
config = {'algorithm': 'ID3'}
cb.fit(pd.read_csv("../dataset/golf2.txt"), config)

print("-------------------------")

print("C4.5 for nominal/numeric features and target:")
config = {'algorithm': 'C4.5'}
cb.fit(pd.read_csv("../dataset/golf2.txt"), config)

print("-------------------------")

print("CART for nominal/numeric features and target:")
config = {'algorithm': 'CART'}
cb.fit(pd.read_csv("../dataset/golf2.txt"), config)

print("-------------------------")

print("regression tree for nominal features, numeric target")
config = {'algorithm': 'Regression'}
cb.fit(pd.read_csv("../dataset/golf3.txt"), config)

print("-------------------------")

print("regression tree for nominal/numeric features, numeric target")
config = {'algorithm': 'Regression'}
cb.fit(pd.read_csv("../dataset/golf4.txt"), config)

print("-------------------------")

print("algorithm must be regression tree for numetic target. set any other algorithm.")
config = {'algorithm': 'ID3'}
cb.fit(pd.read_csv("../dataset/golf4.txt"), config)

print("-------------------------")

print("ID3 for nominal features and target (large data set)")
config = {'algorithm': 'ID3'}
cb.fit(pd.read_csv("../dataset/car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"]), config)

print("-------------------------")

print("C4.5 for nominal features and target (large data set)")
config = {'algorithm': 'C4.5'}
cb.fit(pd.read_csv("../dataset/car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"]), config)

print("-------------------------")

print("CART for nominal features and target (large data set)")
config = {'algorithm': 'CART'}
cb.fit(pd.read_csv("../dataset/car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"]), config)

print("-------------------------")

print("Adaboost")
config = {'algorithm': 'ID3', 'enableAdaboost': True}
cb.fit(pd.read_csv("../dataset/adaboost.txt"), config)

print("-------------------------")

print("Regular GBM")
config = {'algorithm': 'CART', 'enableGBM': True, 'epochs': 10, 'learning_rate': 1}
cb.fit(pd.read_csv("../dataset/golf4.txt"), config)

print("-------------------------")

print("GBM for classification")
config = {'algorithm': 'ID3', 'enableGBM': True, 'epochs': 10, 'learning_rate': 1}
cb.fit(pd.read_csv("../dataset/iris.data", names=["Sepal length","Sepal width","Petal length","Petal width","Decision"]), config)

print("-------------------------")

print("Random forest")
config = {'algorithm': 'ID3', 'enableRandomForest': True, 'num_of_trees': 5, 'enableMultitasking': False}
cb.fit(pd.read_csv("../dataset/car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"]), config)

print("-------------------------")

"""
print("Random forest (multitasking)")
config = {'algorithm': 'ID3', 'enableGBM': False, 'epochs': 10, 'learning_rate': 1, 'enableRandomForest': True, 'num_of_trees': 5, 'enableMultitasking': True, 'enableAdaboost': False}
cb.fit(pd.read_csv("../dataset/car.data",names=["buying","maint","doors","persons","lug_boot","safety","Decision"]), config)
"""
print("-------------------------")
print("unit tests completed successfully...")
