def findDecision(obj): #obj[0]: Sepal length, obj[1]: Sepal width, obj[2]: Petal length, obj[3]: Petal width
   if obj[2]>1.9:
      if obj[1]<=3.4:
         if obj[0]>4.8:
            if obj[3]>0.6:
               return 0.6666666567325592
      elif obj[1]>3.4:
         return -0.3333333432674408
   elif obj[2]<=1.9:
      return -0.3333333432674408
