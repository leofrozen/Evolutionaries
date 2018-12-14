from single_objective import ABC
import numpy as np
from scipy.optimize import rosen

def Schwefel(x):
    y = np.sum(-x*np.sin(np.sqrt(abs(x))))
    return(y)


def rosenbrock(x):
    # teste
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


#minimum = np.array([420.9687,420.9687,420.9687,420.9687,420.9687,420.9687,420.9687,420.9687])
#print (Schwefel(minimum))
#print (schwefel(minimum))


dimension = 30
max_it = 1000
pop_size = 50
q = 5
#func = schwefel
func = rosen
low = -5
high = 5
lb = low
ub = high

teste = [0] *5

b_sol, teste[0] = ABC.optimize_custom(func, lb, ub, pop_size, dimension, max_it, 20)


print(teste[0])