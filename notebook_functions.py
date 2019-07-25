
import math
import numpy as np



def meld_score(x, y, z):
	"""function to compute patient MELD scores; MELD score formula uses Total Bili, INR, and Creatinine"""
    return (((np.log(x)*0.378) + (np.log(y)*1.120)  + (np.log(z)*0.957)+0.643)*10)


"""Created a function to compute patient FIB-4 scores and subsequently added a new column to hold these new values. 
Fib-4 score formula uses age, AST(Aspartate transaminase), Plt (platelet), ALT (Alanine transaminase)"""

def fib_score(w, x, y, z):  
    return (w*x)/(y*math.sqrt(z))