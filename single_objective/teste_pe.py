from single_objective import PE
import numpy as np
from scipy.optimize import rosen

def Schwefel(x):
    y = np.sum(-x*np.sin(np.sqrt(abs(x))))
    return(y)

#minimum = np.array([420.9687,420.9687,420.9687,420.9687,420.9687,420.9687,420.9687,420.9687])
#print (Schwefel(minimum))
#print (schwefel(minimum))


dimension = 30
max_it = 9000
pop_size = 50
q = 5
#func = schwefel
func = rosen
low = -5
high = 5
lb = low
ub = high


r,fit,std = PE.optimize_custom(func, lb, ub, pop_size, dimension, max_it, q)

print(fit)