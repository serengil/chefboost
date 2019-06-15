def findDecision(obj): #obj[0]: Sepal length, obj[1]: Sepal width, obj[2]: Petal length, obj[3]: Petal width
   if obj[2]>1.9:
      if obj[0]>5.4:
         if obj[1]<=2.9:
            if obj[3]>0.6:
               return -0.21194155514240265
         elif obj[1]>2.9:
            if obj[3]>0.6:
               return -0.15536241233348846
      elif obj[0]<=5.4:
         return -0.21194155514240265
   elif obj[2]<=1.9:
      return 0.4238830804824829
