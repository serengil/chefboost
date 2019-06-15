def findDecision(obj): #obj[0]: Sepal length, obj[1]: Sepal width, obj[2]: Petal length, obj[3]: Petal width
   if obj[3]<=1.7:
      if obj[2]>1.9:
         if obj[0]>4.8:
            if obj[1]<=3.4:
               return 0.4238830804824829
      elif obj[2]<=1.9:
         return -0.21194155514240265
   elif obj[3]>1.7:
      if obj[1]<=3.4:
         if obj[0]>4.8:
            if obj[2]>1.9:
               return -0.42231881618499756
      elif obj[1]>3.4:
         return -0.21194155514240265
