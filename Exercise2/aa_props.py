##############
# Exercise 2.7
##############


def isCharged(a):
     if a in ('R','K','E','D','H'):
      return True
     else:
      return False 

def isPositivelyCharged(a):
  if a in ('R','K','H'):
    return True
  else:
    return False

def isNegativelyCharged(a):
  if a in ('D','E'):
    return True
  else:
    return False

def isAromatic(a):
  if a in ('W','Y','F','H'):
    return True
  else:
    return False

def isPolar(a):
  if a in ('R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y'):
    return True
  else:
    return False

def isProline(a):
  if a == 'P':
    return True
  else:
    return False

def containsSulfur(a):
  if a in ('C','M'):
    return True
  else:
    return False
  
def isAcid(a):
  if a in ('D','E'):
    return True
  else:
    return False
  

def isBasic(a):
   if a in ('R','K','H'):
       return True
   else:
       return False
      
def isHydrophobic(a):
    if a in ('A','I','L','M','V','F','Y','W'):
       return True
    else:
       return False