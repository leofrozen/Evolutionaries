import numpy as np
from scipy.optimize import rosen


def rosenbrock(x):
    # teste
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

#
# dimension = 30
# it = 1000
# pop_size = 50
# lb = -5
# ub = 5
# func = rosenbrock
# num_execs = 5
#
# result = [0]*num_execs
# best = [0]*num_execs
# prob_cross = 0.6
# fact= 0.8
# strategy = 2
#
# bounds = [(-5,5)]*30

#    https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

def de(fobj, bounds, mut=0.8, crossp=0.6, popsize=50, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        #yield best, fitness[best_idx]
    return (fitness[best_idx])

#
# for i in range(num_execs):
#    best[i] = (de(rosen,bounds, mut=fact))
#
# print (best)