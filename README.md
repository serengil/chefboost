# ChefBoost

[![Downloads](https://pepy.tech/badge/chefboost)](https://pepy.tech/project/chefboost)
[![Stars](https://img.shields.io/github/stars/serengil/chefboost?color=yellow)](https://github.com/serengil/chefboost)
[![License](http://img.shields.io/:license-MIT-green.svg?style=flat)](https://github.com/serengil/chefboost/blob/master/LICENSE)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dserengil%26type%3Dpatrons&style=flat)](https://www.patreon.com/serengil?repo=chefboost)
[![Twitter](https://img.shields.io/twitter/follow/serengil?color=blue&logo=twitter&style=flat)](https://twitter.com/serengil)
[![DOI](http://img.shields.io/:DOI-10.5281/zenodo.5576203-blue.svg?style=flat)](https://doi.org/10.5281/zenodo.5576203)

**ChefBoost** is a lightweight decision tree framework for Python **with categorical feature support**. It covers regular decision tree algorithms: [ID3](https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/), [C4.5](https://sefiks.com/2018/05/13/a-step-by-step-c4-5-decision-tree-example/), [CART](https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/), [CHAID](https://sefiks.com/2020/03/18/a-step-by-step-chaid-decision-tree-example/) and [regression tree](https://sefiks.com/2018/08/28/a-step-by-step-regression-decision-tree-example/); also some advanved techniques: [gradient boosting](https://sefiks.com/2018/10/04/a-step-by-step-gradient-boosting-decision-tree-example/), [random forest](https://sefiks.com/2017/11/19/how-random-forests-can-keep-you-from-decision-tree/) and [adaboost](https://sefiks.com/2018/11/02/a-step-by-step-adaboost-example/). You just need to write **a few lines of code** to build decision trees with Chefboost.

**Installation** - [`Demo`](https://youtu.be/YYF993HTHf8)

The easiest way to install ChefBoost framework is to download it from [from PyPI](https://pypi.org/project/chefboost). It's going to install the library itself and its prerequisites as well.

```
pip install chefboost
```

Then, you will be able to import the library and use its functionalities

```python
from chefboost import Chefboost as chef
```

**Usage** - [`Demo`](https://youtu.be/Z93qE5eb6eg)

Basically, you just need to pass the dataset as pandas data frame and the optional tree configurations as illustrated below.

```python
import pandas as pd

df = pd.read_csv("dataset/golf.txt")
config = {'algorithm': 'C4.5'}
model = chef.fit(df, config = config, target_label = 'Decision')
```

**Pre-processing**

Chefboost handles the both numeric and nominal features and target values in contrast to its alternatives. So, you don't have to apply any pre-processing to build trees.

**Outcomes**

Built decision trees are stored as python if statements in the `tests/outputs/rules` directory. A sample of decision rules is demonstrated below.

```python
def findDecision(Outlook, Temperature, Humidity, Wind):
   if Outlook == 'Rain':
      if Wind == 'Weak':
         return 'Yes'
      elif Wind == 'Strong':
         return 'No'
      else:
         return 'No'
   elif Outlook == 'Sunny':
      if Humidity == 'High':
         return 'No'
      elif Humidity == 'Normal':
         return 'Yes'
      else:
         return 'Yes'
   elif Outlook == 'Overcast':
      return 'Yes'
   else:
      return 'Yes'
 ```

**Testing for custom instances**

Decision rules will be stored in `outputs/rules/` folder when you build decision trees. You can run the built decision tree for new instances as illustrated below.

```python
prediction = chef.predict(model, param = ['Sunny', 'Hot', 'High', 'Weak'])
```

You can consume built decision trees directly as well. In this way, you can restore already built decision trees and skip learning steps, or apply [transfer learning](https://youtu.be/9hX8ir7_ZtA). Loaded trees offer you findDecision method to test for new instances.

```python
moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
tree = chef.restoreTree(moduleName)
prediction = tree.findDecision(['Sunny', 'Hot', 'High', 'Weak'])
```

tests/global-unit-test.py will guide you how to build a different decision trees and make predictions.

**Model save and restoration**

You can save your trained models. This makes your model ready for transfer learning.

```python
chef.save_model(model, "model.pkl")
```

In this way, you can use the same model later to just make predictions. This skips the training steps. Restoration requires to store .py and .pkl files under `outputs/rules`.

```python
model = chef.load_model("model.pkl")
prediction = chef.predict(model, ['Sunny',85,85,'Weak'])
```

### Sample configurations

ChefBoost supports several decision tree, bagging and boosting algorithms. You just need to pass the configuration to use different algorithms.

**Regular Decision Trees**

Regular decision tree algorithms find the best feature and the best split point maximizing the information gain. It builds decision trees recursively in child nodes.

```python
config = {'algorithm': 'C4.5'} #Set algorithm to ID3, C4.5, CART, CHAID or Regression
model = chef.fit(df, config)
```

The following regular decision tree algorithms are wrapped in the library.

| Algorithm  | Metric | Tutorial | Demo |
| ---        | --- | ---      | ---  |
| ID3        | Entropy, Information Gain |[`Tutorial`](https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/) | [`Demo`](https://youtu.be/Z93qE5eb6eg) |
| C4.5       | Entropy, Gain Ratio | [`Tutorial`](https://sefiks.com/2018/05/13/a-step-by-step-c4-5-decision-tree-example/) | [`Demo`](https://youtu.be/kjhQHmtDaAA) |
| CART       | GINI | [`Tutorial`](https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/) | [`Demo`](https://youtu.be/CSApBetgukM) |
| CHAID      | Chi Square | [`Tutorial`](https://sefiks.com/2020/03/18/a-step-by-step-chaid-decision-tree-example/) | [`Demo`](https://youtu.be/dcnFuS4QILg) |
| Regression | Standard Deviation | [`Tutorial`](https://sefiks.com/2018/08/28/a-step-by-step-regression-decision-tree-example/) | [`Demo`](https://youtu.be/pCQ2RCa20Bg) |

**Gradient Boosting** [`Tutorial`](https://sefiks.com/2018/10/04/a-step-by-step-gradient-boosting-decision-tree-example/), [`Demo`](https://youtu.be/KFsnZKMKNAE)

Gradient boosting is basically based on building a tree, and then building another based on the previous one's error. In this way, it boosts results. Predictions will be the sum of each tree'e prediction result.

```python
config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1, 'max_depth': 5}
```

**Random Forest** [`Tutorial`](https://sefiks.com/2017/11/19/how-random-forests-can-keep-you-from-decision-tree/), [`Demo`](https://youtu.be/J7hDtV261PQ)

Random forest basically splits the data set into several sub data sets and builds different data set for those sub data sets. Predictions will be the average of each tree's prediction result.

```python
config = {'enableRandomForest': True, 'num_of_trees': 5}
```

**Adaboost** [`Tutorial`](https://sefiks.com/2018/11/02/a-step-by-step-adaboost-example/), [`Demo`](https://youtu.be/Obj208F6e7k)

Adaboost applies a decision stump instead of a decision tree. This is a weak classifier and aims to get min 50% score. It then increases the unclassified ones and decreases the classified ones. In this way, it aims to have a high score with weak classifiers.

```python
config = {'enableAdaboost': True, 'num_of_weak_classifier': 4}
```

**Feature Importance** - [`Demo`](https://youtu.be/NFLQT6Ta4-k)

Decision trees are naturally interpretable and explainable algorithms. A decision is clear made by a single tree. Still we need some extra layers to understand the built models. Besides, random forest and GBM are hard to explain. Herein, [feature importance](https://sefiks.com/2020/04/06/feature-importance-in-decision-trees/) is one of the most common way to see the big picture and understand built models.

```python
df = chef.feature_importance("outputs/rules/rules.py")
```

| feature     | final_importance |
| ---         | ---              |
| Humidity    | 0.3688           |
| Wind        | 0.3688           |
| Outlook     | 0.2624           |
| Temperature | 0.0000           |

### Paralellism

ChefBoost offers parallelism to speed model building up. Branches of a decision tree will be created in parallel in this way. You should set enableParallelism argument to True in the configuration. Its default value is False. It allocates half of the total number of cores in your environment if parallelism is enabled.

```python
if __name__ == '__main__':
   config = {'algorithm': 'C4.5', 'enableParallelism': True, 'num_cores': 2}
   model = chef.fit(df, config)
```

Notice that you have to locate training step in an if block and it should check you are in main.

### Contributing

Pull requests are welcome. You should run the unit tests locally by running [`test/global-unit-test.py`](https://github.com/serengil/chefboost/blob/master/tests/global-unit-test.py). Please share the unit test result logs in the PR.

### Support

There are many ways to support a project - starring⭐️ the GitHub repos is just one 🙏

You can also support this work on [Patreon](https://www.patreon.com/serengil?repo=chefboost)

<a href="https://www.patreon.com/serengil?repo=chefboost">
<img src="https://raw.githubusercontent.com/serengil/chefboost/master/icon/patreon.png" width="30%" height="30%">
</a>

### Citation

Please cite [ChefBoost](https://doi.org/10.5281/zenodo.5576203) in your publications if it helps your research. Here is an example BibTeX entry:

```BibTeX
@misc{serengil2021chefboost,
  author       = {Serengil, Sefik Ilkin},
  title        = {ChefBoost: A Lightweight Boosted Decision Tree Framework},
  month        = oct,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5576203},
  howpublished = {https://doi.org/10.5281/zenodo.5576203}
}
```

Also, if you use chefboost in your GitHub projects, please add chefboost in the requirements.txt.

### Licence

ChefBoost is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/chefboost/blob/master/LICENSE) for more details.
