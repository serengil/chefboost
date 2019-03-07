# chefboost

<p align="center"><img src="https://raw.githubusercontent.com/serengil/chefboost/master/icon/chefboost.jpg" width="200" height="200"></p>

Chefboost is a gradient boosting and random forest enabled decision tree framework. ID3, C4.5, CART and regression tree algorithms are supported.

# Usage

Basically, you just need to pass the dataset as pandas data frame and tree configurations after importing Chefboost as illustrated below. You just need to set the label of the target column to **"Decision"**. 

```
import Chefboost as chef
import pandas as pd

#GBM example

config = {
	'algorithm': 'ID3' #ID3, C4.5, CART, Regression
	, 'enableGBM': False, 'epochs': 10, 'learning_rate': 1
	, 'enableRandomForest': True, 'num_of_trees': 5, 'enableMultitasking': True
	, 'enableAdaboost': False, 'debug': False
}

df = pd.read_csv("dataset/golf3.txt")

chef.fit(df, config)
```

Initial tests are run on Python 3.6.4 and Windows 10 OS.

# Outcomes

Built decision trees are stored as python if statements in the outputs/rules directory. A sample of decision rules is demonstrated below.

```
def findDecision(Outlook,Temperature,Humidity,Wind,Decision):
   if Outlook == 'Rain':
      if Wind == 'Weak':
         return 'Yes'
      if Wind == 'Strong':
         return 'No'
   if Outlook == 'Sunny':
      if Humidity == 'High':
         return 'No'
      if Humidity == 'Normal':
         return 'Yes'
   if Outlook == 'Overcast':
      return 'Yes'
 ```

# Prerequisites

Pandas and numpy python libraries are used to load data sets in this repository. You might run the following commands to install these packages if you are going to use them first time.

```
pip install pandas
pip install numpy
```

# Documentation

You can find detailed documentations about these core algorithms [here](https://sefiks.com/tag/decision-tree/).

# Licence

You can use, clone or distribute any content of this repository just to the extent that you cite or reference.
