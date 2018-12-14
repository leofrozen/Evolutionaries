import numpy as np
import random


def init_population(func, lb, ub, pop_size, dimension):
    pop = np.random.uniform(low=lb, high=ub, size=(pop_size, dimension))
    fitness = np.full(pop_size, np.nan)
    fitness = np.apply_along_axis(func, 1, pop)
    return (pop, fitness)



def simple_crossover(pop_size, dimension, tmp_pop, prob_cross):
    offsprings = np.full((pop_size, dimension), np.nan)

    cross_point = int(dimension/2)
    indexs = random.sample(range(pop_size), int(pop_size*prob_cross))

    its = (len(indexs) - 1)

    for i in range(0, its, 2): # i is a random index  between 0 ~ pop_size -1

        parent_1 = tmp_pop[indexs[i]]
        parent_2 = tmp_pop[indexs[i+1]]

        offspring_1 = np.concatenate((parent_1[:cross_point], [parent_2[cross_point:]]), axis=None)
        offspring_2 = np.concatenate((parent_2[:cross_point], [parent_1[cross_point:]]), axis=None)

        tmp_pop[indexs[i]] = offspring_1
        tmp_pop[indexs[i+1]] = offspring_2


    return tmp_pop



def simple_crossover2(pop_size, dimension, tmp_pop, prob_cross):
    offsprings = np.full((pop_size, dimension), np.nan)

    cross_point = int(dimension/2)
    indexs = random.sample(range(pop_size), int(pop_size*prob_cross))

    its = (len(indexs) - 1)

    for i in range(0, its, 2): # i is a random index  between 0 ~ pop_size -1
        #print(i)
        parent_1 = tmp_pop[indexs[i]]
        parent_2 = tmp_pop[indexs[i+1]]

        offspring_1 = np.concatenate((parent_1[:cross_point], [parent_2[cross_point:]]), axis=None)
        offspring_2 = np.concatenate((parent_2[:cross_point], [parent_1[cross_point:]]), axis=None)

        tmp_pop[indexs[i]] = offspring_1
        tmp_pop[indexs[i+1]] = offspring_2
    return tmp_pop



def uniform_mutation(func, tmp_pop, pop_size, lb, ub, dimension, prob_muta):
    tmp_fitness = np.random.uniform(lb, ub, size=(pop_size, dimension))


    num_targets = random.randint(1,int(0.5*pop_size)) # random entre 1 e 20% da população
    indexs = random.sample(range(pop_size), num_targets)  # take a random row and do the mutation

    for i in indexs:
        #tmp_pop[i, random.randrange(0, dimension)] = random.randrange(lb,ub)  # the choosen line, a random column: receive a random value between lb and ub
        tmp_pop[i, random.randrange(0, dimension)] = random.uniform(lb,ub)  # the choosen line, a random column: receive a random value between lb and ub

    return tmp_pop



def selection(pop, pop_size, dimension, fitness, sel, t_size):
    if sel == 1:
        new_pop = roulette(pop, fitness, pop_size, dimension)
        return new_pop
    elif sel == 2:
        new_pop = tournament(pop, fitness, pop_size, dimension, t_size)
        return new_pop


def tournament(pop, fitness, pop_size, dimension, t_size):
    new_pop = np.full((pop_size,dimension), np.nan)

    for i in range(pop_size):
        indexs = random.sample(range(pop_size), t_size)

        pos = fitness[indexs].argmin()

        new_pop[i,] = pop[indexs[pos],]
    return new_pop

def roulette(pop, fitness, pop_size, dimension):
    pass
    # new_pop = np.NAN(size=(pop_size, dimension))
    # new_fit = np.NAN(pop_size)
    # a = sum(fitness)
    # # p = [x/a for x in fitness]
    # p = -fitness / a
    # q = np.cumsum(p)
    # # r = randint(0,pop_size)
    # r = np.random.uniform()
    # for i in range(pop_size):
    #     if r[i] < q[1]:
    #         new_pop[i,] = pop[1,]
    #     else:


def ga(func, lb, ub, pop_size, dimension, iterations, prob_cross=0.6, prob_muta=0.01, select_type=2, t_size = 4, elitism = True):
    tmp_pop = np.full((pop_size, dimension), np.nan)
    pop, fitness = init_population(func, lb, ub, pop_size, dimension)

    for i in range(iterations):
        tmp_pop = selection(pop,pop_size,dimension,fitness,select_type,t_size)
        tmp_pop = simple_crossover(pop_size,dimension,tmp_pop,prob_cross)
        if random.randint(0, 100) <= prob_muta * 100:  # prob_muta % chance to start a mutation
            tmp_pop = uniform_mutation(func, tmp_pop, pop_size, lb, ub, dimension, prob_muta)
        # nova avaliacao
        tmp_fitness = np.apply_along_axis(func, 1, tmp_pop)


        if (elitism):
            best_tmp = np.min(tmp_fitness)
            best_old = np.min(fitness)
            if (best_old < best_tmp):
                index = fitness.argmin()
                index_worst = tmp_fitness.argmax()
                tmp_pop[index_worst,] = pop[index,]
                tmp_fitness[index_worst,] = fitness[index,]

        pop = tmp_pop
        fitness = tmp_fitness

    return (pop[fitness.argmin()], np.min(fitness))






###################################################


def ed_strategy(pop, pop_size, fitness, strategy=1):
    indexs = None
    if strategy == 1:
        indexs = random.sample(range(pop_size), 3)
    elif strategy == 2:
        indexs = random.sample(range(pop_size), 2)
        best = fitness.argmin()
        indexs.insert(0,best)

    return indexs

# def ed_mutation(factor, indexs, pop, pop_size):
#     #pop[indexs[0]] + factor * (pop[1] + pop[2])
#     #return
#     return (pop[indexs[0]] + factor*(pop[1] + pop[2]))
#

#
# def cut_upper_border(arr, border):
#     arr[arr > border] = border


def ed(func,lb, ub, pop_size, dimension, iterations, factor, prob_cross=0.6, strategy = 1 ):
    #tmp_pop = np.full((pop_size, dimension), np.nan)
    pop, fitness = init_population(func, lb, ub, pop_size, dimension)

    for i in range(iterations):
        for j in range(pop_size):

            indexs = ed_strategy(pop,pop_size, fitness, strategy) # pega 3 indices
            new_sol = pop[indexs[0]] + factor*(pop[1] + pop[2]) # mutacao

            # cut BORDERS
            #print("cut_borders")
            #print (new_sol)

            # new_sol[new_sol > ub] = np.random.uniform(low=lb,high=ub, size=1)[0]
            # new_sol[new_sol < lb] = np.random.uniform(low=lb,high=ub, size=1)[0]

            #print(new_sol)

            for k in range(dimension):  ## CROSSOVER

                if new_sol[k] > ub or new_sol[k] < lb: new_sol[k] = pop[indexs[0],k]
                if random.randint(0, 100) <= prob_cross * 100: # probabilidade de cruzar
                    new_sol[k] = pop[indexs[0],k]

            new_fit = func(new_sol)

            if new_fit < fitness[indexs[0]]: # MINIZACAO
                pop[indexs[0]] = new_sol
                fitness[indexs[0]] = new_fit

    return (pop[fitness.argmin()], np.min(fitness))

