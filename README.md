# chefboost

<p align="center"><img src="https://raw.githubusercontent.com/serengil/chefboost/master/icon/chefboost.jpg" width="200" height="200"></p>

**Chefboost** is a lightweight [gradient boosting](https://sefiks.com/2018/10/04/a-step-by-step-gradient-boosting-decision-tree-example/), [random forest](https://sefiks.com/2017/11/19/how-random-forests-can-keep-you-from-decision-tree/) and [adaboost](https://sefiks.com/2018/11/02/a-step-by-step-adaboost-example/) enabled decision tree framework including regular [ID3](https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/), [C4.5](https://sefiks.com/2018/05/13/a-step-by-step-c4-5-decision-tree-example/), [CART](https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/) and [regression tree](https://sefiks.com/2018/08/28/a-step-by-step-regression-decision-tree-example/) algorithms **with categorical features support**. It is lightweight, you just need to write **a few lines of code** to build decision trees with Chefboost.

# Usage

Basically, you just need to pass the dataset as pandas data frame and tree configurations after importing Chefboost as illustrated below. You just need to put the target label to the right. Besides, chefboost handles both numeric and nominal features and target values in contrast to its alternatives.

```python
import Chefboost as chef
import pandas as pd

df = pd.read_csv("dataset/golf.txt")

config = {'algorithm': 'ID3'}
model = chef.fit(df, config)
```

# Outcomes

Built decision trees are stored as python if statements in the `outputs/rules/rules.py` file. A sample of decision rules is demonstrated below.

```python
def findDecision(Outlook,Temperature,Humidity,Wind,Decision):
   if Outlook == 'Rain':
      if Wind == 'Weak':
         return 'Yes'
      elif Wind == 'Strong':
         return 'No'
   elif Outlook == 'Sunny':
      if Humidity == 'High':
         return 'No'
      elif Humidity == 'Normal':
         return 'Yes'
   elif Outlook == 'Overcast':
      return 'Yes'
 ```

# Testing for custom instances

Decision rules will be stored in `outputs/rules/` folder when you build decision trees. You can run the built decision tree for new instances as illustrated below.

```python
test_instance = ['Sunny', 'Hot', 'High', 'Weak']
model = chef.fit(df, config)
prediction = chef.predict(model, test_instance)
```

You can consume built decision trees directly as well. In this way, you can restore already built decision trees and skip learning steps, or apply **transfer learning**. Loaded trees offer you findDecision method to test for new instances.

```python
from commons import functions
moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
tree = functions.restoreTree(moduleName)
prediction = tree.findDecision(['Sunny', 'Hot', 'High', 'Weak'])
```

**Dispathcher.py** will guide you how to build a different decision trees and make predictions.

# Model save and restoration

You can save your trained models.

```python
model = chef.fit(df.copy(), config)
chef.save_model(model, "model.pkl")
```

In this way, you can use the same model later to just make predictions. This skips the training steps. Restoration requires to store .py and .pkl files under `outputs/rules`.

```python
model = chef.load_model("model.pkl")
prediction = chef.predict(model, ['Sunny',85,85,'Weak'])
```

# Sample configurations

Chefboost supports several decision tree, bagging and boosting algorithms. You just need to pass the configuration to use different algorithms.

**Regular Decision Trees** [`ID3 Video`](https://youtu.be/Z93qE5eb6eg), [`C4.5 Video`](https://youtu.be/kjhQHmtDaAA), [`CART Video`](https://youtu.be/CSApBetgukM), [`Regression Tree Video`](https://youtu.be/pCQ2RCa20Bg)

```python
config = {'algorithm': 'C4.5'} #ID3, C4.5, CART or Regression
```

**Gradient Boosting** [`Video`](https://youtu.be/KFsnZKMKNAE)

```python
config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1}
```

**Random Forest** [`Video`](https://youtu.be/J7hDtV261PQ)

```python
config = {'enableRandomForest': True, 'num_of_trees': 5}
```

**Adaboost** [`Video`](https://youtu.be/Obj208F6e7k)

```python
config = {'enableAdaboost': True, 'num_of_weak_classifier': 4}
```

# Prerequisites

Pandas and numpy python libraries are used to load data sets in this repository. You might run the following commands to install these packages if you are going to use them first time.

```
pip install pandas==0.22.0
pip install numpy==1.14.0
pip install tqdm==4.30.0
```

Initial tests are run on the following environment.

 ```
C:\>python --version
Python 3.6.4 :: Anaconda, Inc.
 ```
 
# Documentation

This YouTube [playlist](https://www.youtube.com/playlist?list=PLsS_1RYmYQQHp_xZObt76dpacY543GrJD) guides you how to use Chefboost step by step for different algorithms. You can also find the detailed documentations about these core algorithms [here](https://sefiks.com/tag/decision-tree/). 

Besides, you can enroll this online course - [**Decision Trees for Machine Learning From Scratch**](https://www.udemy.com/course/decision-trees-for-machine-learning/?referralCode=FDC9B836EC6DAA1A663A) and follow the curriculum if you wonder the theory of decision trees and how this framework is developed.

# Support

There are many ways to support a project - starring the GitHub repos is one.

# Licence

Chefboost is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/chefboost/blob/master/LICENSE) for more details.
