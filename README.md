# chefboost

Chefboost is a gradient boosting and random forest enabled decision tree framework. ID3, C4.5, CART and regression tree algorithms are supported.

# Usage

Basically, you just need to pass the dataset as pandas data frame and tree configurations after importing Chefboost as illustrated below.

```
import Chefboost as cb
import pandas as pd

#GBM example

config = {
	'algorithm': 'ID3' #ID3, C4.5, CART, Regression
	, 'enableGBM': False, 'epochs': 10, 'learning_rate': 1
	, 'enableRandomForest': True, 'num_of_trees': 5, 'enableMultitasking': True
	, 'enableAdaboost': False, 'debug': False
}

df = pd.read_csv("dataset/golf3.txt")

cb.fit(df, config)
```

# Prerequisites

Pandas and numpy python libraries are used to load data sets in this repository. You might run the following commands to install these packages if you are going to use them first time.

```
pip install pandas
pip install numpy
```

# Documentation

You can find detailed documentations about these core algorithms [here](https://sefiks.com/tag/decision-tree/).
