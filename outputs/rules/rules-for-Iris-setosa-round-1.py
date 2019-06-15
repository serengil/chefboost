def findDecision(obj): #obj[0]: Sepal length, obj[1]: Sepal width, obj[2]: Petal length, obj[3]: Petal width
   if obj[2]>1.9:
      return -0.3333333432674408
   elif obj[2]<=1.9:
      return 0.6666666567325592
