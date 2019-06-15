def findDecision(obj): #obj[0]: Sepal length, obj[1]: Sepal width, obj[2]: Petal length, obj[3]: Petal width
   if obj[2]>1.9:
      if obj[1]<=3.4:
         if obj[0]>4.8:
            if obj[3]>0.6:
               return 0.48042434453964233
      elif obj[1]>3.4:
         return -0.24850180745124817
   elif obj[2]<=1.9:
      return -0.24021212756633759
