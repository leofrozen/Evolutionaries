import numpy as np
import random



def init_population(func, lb, ub, v_min, v_max, swarm_size, dimension):
    swarm = np.random.uniform(low=lb, high=ub, size=(swarm_size, dimension))
    #fitness = np.full(swarm_size, np.nan)
    fitness = np.apply_along_axis(func, 1, swarm)
    velocity = np.random.uniform(low=v_min, high=v_max, size=(swarm_size, dimension))
    #velocity = velocity*(v_max - v_min) + v_min
    return (swarm, fitness, velocity)


def ajust_bounds(*args):
    # x, lb = -5, ub = 5, rand = False
    x = args[0]
    lb = args[1]
    ub = args[2]
    rand = args[3]

    higher = x > ub
    lower = x < lb

    if rand:
        mean_border = (ub + lb)/2
        x[higher] = random.uniform(lb,ub)
        x[lower] = random.uniform(lb,ub)
    else:
        x[higher] = ub
        x[lower] = lb
    return x

def optimize_custom(func, lb, ub, v_min, v_max, swarm_size, dimension, iterations,w_min,w_max, c1,c2):
    #S = np.full((swarm_size,dimension), np.nan)
    #P = np.full((swarm_size,dimension), np.nan)
    #V = np.full((swarm_size,dimension), np.nan)
    #fitness = np.full(swarm_size, np.nan)
    S, fitness, V = init_population(func, lb, ub, v_min, v_max, swarm_size, dimension)

    #idx =
    g = S[fitness.argmin()]
    g_fit = np.min(fitness)
    #print(g)
    #print (g_fit)

    P = S.copy()                     # WHY ???
    p_fitness = fitness


    decaimento = False
    if w_max > w_min:
        decaimento = True
    w = w_max
    testes = 0
    for i in range(iterations):
        r1 = np.random.rand(swarm_size,dimension)
        r2 = np.random.rand(swarm_size,dimension)

        V = V*w + c1*r1*(P-S) + c2*r2*(g-S)
        S = S + V

        S = np.apply_along_axis(ajust_bounds, 1, S,-5,5,False)

        fitness = np.apply_along_axis(func, 1, S)

        if g_fit > np.min(fitness):
            g_fit = np.min(fitness)
            g = S[fitness.argmin()]
        indexs = fitness < p_fitness
        P[indexs] = S[indexs]
        p_fitness[indexs] = fitness[indexs]

        if decaimento:
            w = (w_max-w_min) * (iterations-i)/iterations + w_min


        # if testes == 100:
        #     print (np.min(p_fitness))
        #     testes = 0
        # testes += 1

    return (g, g_fit)











