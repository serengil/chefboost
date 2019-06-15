def findDecision(obj): #obj[0]: buying, obj[1]: maint, obj[2]: doors, obj[3]: persons, obj[4]: lug_boot, obj[5]: safety
   if obj[5] == 'med':
      if obj[3] == '4':
         if obj[0] == 'low':
            if obj[1] == 'high':
               return 'acc'
            elif obj[1] == 'low':
               if obj[2] == '5more':
                  return 'good'
               elif obj[2] == '2':
                  return 'acc'
               elif obj[2] == '3':
                  return 'acc'
               elif obj[2] == '4':
                  return 'good'
            elif obj[1] == 'vhigh':
               return 'unacc'
            elif obj[1] == 'med':
               if obj[2] == '5more':
                  return 'good'
               elif obj[2] == '3':
                  return 'acc'
         elif obj[0] == 'high':
            if obj[4] == 'small':
               return 'unacc'
            elif obj[4] == 'med':
               if obj[2] == '5more':
                  return 'acc'
               elif obj[2] == '3':
                  return 'unacc'
               elif obj[2] == '4':
                  return 'acc'
            elif obj[4] == 'big':
               if obj[1] == 'vhigh':
                  return 'unacc'
               elif obj[1] == 'high':
                  return 'acc'
         elif obj[0] == 'vhigh':
            if obj[1] == 'high':
               return 'unacc'
            elif obj[1] == 'med':
               if obj[4] == 'big':
                  return 'acc'
               elif obj[4] == 'small':
                  return 'unacc'
            elif obj[1] == 'vhigh':
               return 'unacc'
            elif obj[1] == 'low':
               return 'unacc'
         elif obj[0] == 'med':
            if obj[4] == 'med':
               return 'acc'
            elif obj[4] == 'big':
               return 'acc'
            elif obj[4] == 'small':
               if obj[1] == 'low':
                  return 'acc'
               elif obj[1] == 'high':
                  return 'unacc'
      elif obj[3] == 'more':
         if obj[4] == 'small':
            if obj[1] == 'low':
               if obj[0] == 'vhigh':
                  return 'unacc'
               elif obj[0] == 'high':
                  return 'unacc'
               elif obj[0] == 'med':
                  return 'unacc'
               elif obj[0] == 'low':
                  return 'acc'
            elif obj[1] == 'high':
               if obj[2] == '2':
                  return 'unacc'
               elif obj[2] == '4':
                  return 'acc'
               elif obj[2] == '5more':
                  return 'unacc'
            elif obj[1] == 'vhigh':
               return 'unacc'
            elif obj[1] == 'med':
               return 'acc'
         elif obj[4] == 'big':
            if obj[1] == 'med':
               if obj[0] == 'high':
                  return 'acc'
               elif obj[0] == 'med':
                  return 'acc'
               elif obj[0] == 'vhigh':
                  return 'acc'
               elif obj[0] == 'low':
                  return 'good'
            elif obj[1] == 'vhigh':
               return 'acc'
            elif obj[1] == 'low':
               if obj[0] == 'med':
                  return 'good'
               elif obj[0] == 'vhigh':
                  return 'acc'
               elif obj[0] == 'low':
                  return 'good'
            elif obj[1] == 'high':
               return 'acc'
         elif obj[4] == 'med':
            if obj[1] == 'high':
               if obj[2] == '4':
                  return 'acc'
               elif obj[2] == '2':
                  return 'unacc'
               elif obj[2] == '5more':
                  return 'acc'
            elif obj[1] == 'med':
               if obj[0] == 'vhigh':
                  if obj[2] == '4':
                     return 'acc'
                  elif obj[2] == '2':
                     return 'unacc'
               elif obj[0] == 'med':
                  return 'acc'
            elif obj[1] == 'low':
               return 'good'
            elif obj[1] == 'vhigh':
               return 'acc'
      elif obj[3] == '2':
         return 'unacc'
   elif obj[5] == 'low':
      return 'unacc'
   elif obj[5] == 'high':
      if obj[3] == '2':
         return 'unacc'
      elif obj[3] == '4':
         if obj[4] == 'big':
            if obj[0] == 'med':
               if obj[1] == 'vhigh':
                  return 'acc'
               elif obj[1] == 'high':
                  return 'acc'
               elif obj[1] == 'med':
                  return 'vgood'
            elif obj[0] == 'low':
               return 'vgood'
            elif obj[0] == 'vhigh':
               return 'acc'
            elif obj[0] == 'high':
               return 'acc'
         elif obj[4] == 'med':
            if obj[2] == '5more':
               if obj[0] == 'low':
                  return 'vgood'
               elif obj[0] == 'vhigh':
                  return 'unacc'
               elif obj[0] == 'med':
                  return 'vgood'
            elif obj[2] == '4':
               if obj[0] == 'vhigh':
                  return 'acc'
               elif obj[0] == 'high':
                  return 'unacc'
               elif obj[0] == 'med':
                  return 'acc'
            elif obj[2] == '2':
               if obj[0] == 'low':
                  return 'acc'
               elif obj[0] == 'high':
                  return 'unacc'
            elif obj[2] == '3':
               return 'acc'
         elif obj[4] == 'small':
            if obj[0] == 'low':
               if obj[1] == 'vhigh':
                  return 'acc'
               elif obj[1] == 'high':
                  return 'acc'
               elif obj[1] == 'med':
                  return 'good'
               elif obj[1] == 'low':
                  return 'good'
            elif obj[0] == 'vhigh':
               return 'acc'
            elif obj[0] == 'med':
               if obj[1] == 'low':
                  return 'good'
               elif obj[1] == 'med':
                  return 'acc'
            elif obj[0] == 'high':
               return 'acc'
      elif obj[3] == 'more':
         if obj[0] == 'med':
            if obj[1] == 'vhigh':
               return 'acc'
            elif obj[1] == 'low':
               if obj[2] == '4':
                  return 'vgood'
               elif obj[2] == '2':
                  return 'vgood'
               elif obj[2] == '5more':
                  return 'good'
            elif obj[1] == 'high':
               return 'acc'
            elif obj[1] == 'med':
               return 'vgood'
         elif obj[0] == 'high':
            if obj[1] == 'vhigh':
               return 'unacc'
            elif obj[1] == 'low':
               return 'acc'
            elif obj[1] == 'high':
               return 'unacc'
            elif obj[1] == 'med':
               return 'acc'
         elif obj[0] == 'low':
            if obj[1] == 'vhigh':
               return 'acc'
            elif obj[1] == 'med':
               if obj[2] == '2':
                  if obj[4] == 'med':
                     return 'good'
                  elif obj[4] == 'small':
                     return 'unacc'
               elif obj[2] == '3':
                  return 'vgood'
            elif obj[1] == 'high':
               return 'acc'
         elif obj[0] == 'vhigh':
            if obj[1] == 'med':
               return 'acc'
            elif obj[1] == 'low':
               if obj[2] == '4':
                  return 'acc'
               elif obj[2] == '2':
                  return 'unacc'
            elif obj[1] == 'vhigh':
               return 'unacc'
