import numpy as np
import random


# links: 
# http://www.scholarpedia.org/article/Artificial_bee_colony_algorithm#Eq-5
# https://github.com/rwuilbercq/Hive/blob/master/Hive/Hive.py
# http://cleverowl.uk/2015/07/01/using-one-way-anova-and-tukeys-test-to-compare-data-sets/

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



def init_population(func, lb, ub, pop_size, dimension):
    pop = np.random.uniform(low=lb, high=ub, size=(pop_size, dimension))
    fitness = np.apply_along_axis(func, 1, pop).reshape(pop_size,1)
    #
    # fit2 = np.apply_along_axis(func, 1, pop).reshape(len(fitness), 1)
    #
    # print("min fit2: ", int(np.min(fit2)))
    # print("min fit emp: ", int(np.min(fitness)))

    return (pop, fitness)


def select (pop, fitness, pop_size, dimension):
    #indexs = roulette_wheel(pop, fitness, pop_size, dimension)
    indexs = roulette_eggs(pop, fitness, pop_size, dimension)

    return (indexs)


# def roulette_wheel(pop, fitness, pop_size, dimension):
#     new_pop = np.full((pop_size, dimension), np.nan)
#     #b_indexs = fitness <= fitness.mean()
#     a = fitness.sum()
#     p = fitness / a
#     q = p.cumsum()
#     r = np.random.rand(pop_size)
#     for i in range(pop_size):
#         if ( r[i] < q[i] ):
#             new_pop[i] = pop[i]
#         else:
#             idx = np.min(np.argwhere(q < r[i]))
#             new_pop[i] = pop[idx]
#
#     return (new_pop)


# pegar as X's abelhas com fit ruim e dá a elas a chance de pegar um fit bom aleatoriamente
def roulette_eggs(pop, fitness, pop_size, num):
    b_indexs = np.argwhere(fitness <= fitness.mean())
    # w_indexs = np.argwhere(fitness > fitness.mean())
    # random_b = random.sample(range(len(b_indexs)), min(num,len(b_indexs)))
    # random_w = random.sample(range(len(w_indexs)), min(num,len(w_indexs)))
    #
    # for i in random_w:
    #


    #print(b_indexs)
    return (b_indexs)


def send_employee(func, lb, ub, pop, fitness, dimension):
    new_pop = pop.copy()
    new_pop = np.apply_along_axis(mutation_bee, 1, new_pop,lb,ub,dimension)
    # print (new_pop)
    new_fit = np.apply_along_axis(func, 1, new_pop).reshape(len(fitness),1)

    b_bees = np.argwhere(fitness < new_fit)
    #print (b_bees)
    new_pop[b_bees] = pop[b_bees]
    new_fit[b_bees] = fitness[b_bees]
    # print ("min: ", np.min(new_fit))
    # new_fit = np.apply_along_axis(func,1,new_pop)
    # print("min: ", np.min(new_fit))

    return (new_pop, new_fit)

def mutation_bee(*args):
    bee = args[0]
    lb = args[1]
    ub = args[2]
    dimension = args[3]
    num = int(dimension * 0.05)
    num =1
    mut_indexs = random.sample(range(dimension), num)
    new_bee = bee.copy()
    for i in mut_indexs:
        new_bee[i] = bee[i] + random.uniform(lb,ub)*random.random()
    return (new_bee)


def send_onlookers(pop, fitness, trials, func):
    w_indexs = np.argwhere(fitness > fitness.mean())
    b_indexs = np.argwhere(fitness <= fitness.mean())
    new_pop = pop.copy()
    pop_size = int(len(fitness))
    meio_size = int(pop_size/2)
    terco_size = int(pop_size/3)
    arr_idx = np.arange(pop_size).reshape(pop_size,1)
    merge = np.concatenate((arr_idx,fitness),axis=1)
    merge = merge[np.argsort(merge[:, 1])][::-1]
    for i in range(meio_size):
        num_rand = random.randrange(terco_size*2, pop_size)
        new_pop[int(merge[i,0])] = pop[int(merge[num_rand,0])]
        trials[i] = 0
    new_fit = np.apply_along_axis(func, 1, new_pop).reshape(pop_size,1)
    return (new_pop, trials, new_fit)



def send_scout(pop, lb, ub, dimension, fitness, func, trials, limit):
    tired = np.argwhere(trials > limit)
    if len(tired) > 0:
        #print (trials[0])

        num = max(1, int(len(tired) / 2))
        num = len(tired)
        random_tired = random.sample(range(len(tired)),num)
        for i in random_tired:

            # bee = pop[int(tired[i])].copy()
            # bee += random.uniform(lb,ub)*random.random()
            # bee = ajust_bounds(bee, lb, ub, False)

            bee = np.random.uniform(low=lb, high=ub, size=dimension)


            #if func(bee) < fitness[int(tired[i])]:
            if func(bee) < np.mean(fitness):
                pop[tired[i]] = bee
                trials[tired[i]] = 0
    return (pop, trials)


def optimize_custom(func, lb, ub, pop_size, dimension, interactions, limit):

    pop, fitness = init_population(func, lb, ub, pop_size, dimension)
    trials = np.zeros((pop_size,), dtype=int)
    testes = 0
    new_pop = np.full((pop_size, dimension), np.nan)
    for i in range(interactions):

        fit2 = np.apply_along_axis(func, 1, pop).reshape(len(fitness), 1)

        # send the employed bees
        new_pop, new_fitness = send_employee(func, lb, ub, pop, fitness, dimension)

        # send the onlooker bees
        new_pop, trials, new_fitness = send_onlookers(new_pop, new_fitness, trials, func)
        # send scout bees
        new_pop, trials = send_scout(new_pop, lb, ub, dimension, new_fitness, func, trials, limit)

        fitness = np.apply_along_axis(func, 1, pop)

        pop = new_pop
        fitness = np.apply_along_axis(func, 1, pop).reshape(pop_size,1)
        trials = trials + 1

        # if testes == 50:
        #     print (np.mean(fitness))
        #     testes = 0
        # testes += 1
        #
    best_result = np.argmin(fitness)
    return (pop[best_result], np.min(fitness[0]))
