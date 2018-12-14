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
        #mean_border = (ub + lb)/2
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




def send_employee(func, lb, ub, pop, fitness, dimension, trials, limit):
    new_pop = pop.copy()
    # print (len(new_pop))
    # print (len(trials))
    new_pop = np.concatenate((new_pop, np.array(trials).reshape(len(trials),1)), axis=1) # adicionar o trials
    new_pop = np.apply_along_axis(mutation_bee, 1, new_pop,lb,ub,dimension, limit) # aplica a mutacao
    #trials = new_pop[:,-1] # recupera o trials
    new_pop = np.delete(new_pop,-1,1) # deleta a ultima coluna: o trials
    new_pop = np.apply_along_axis(ajust_bounds, 1, new_pop, lb, ub, True)
    new_fit = np.apply_along_axis(func, 1, new_pop).reshape(len(fitness),1)

    b_bees = np.argwhere(fitness < new_fit)
    #print (b_bees)
    new_pop[b_bees] = pop[b_bees]
    new_fit[b_bees] = fitness[b_bees]


    #return (new_pop, new_fit, trials)
    return (new_pop, new_fit)

def mutation_bee(*args):
    bee = args[0]
    lb = args[1]
    ub = args[2]
    dimension = args[3]
    limite = args[4]

    num = 1
    if bee[-1] > limite:
        num = max(int(dimension * 0.5), 1)
        #bee[-1] = 0 # zera o trials novamente

    mut_indexs = random.sample(range(dimension), num)
    new_bee = bee.copy()

    for i in mut_indexs:
        # new_bee[i] = random.uniform(lb, ub)
        new_bee[i] = bee[i] + random.uniform(lb, ub) * random.random()
    # if random.randint(0,50) == 0:
    #     for i in mut_indexs:
    #         #new_bee[i] = random.uniform(lb, ub)
    #         new_bee[i] = bee[i] + random.uniform(lb, ub) * random.random()
    # else:
    #     for i in mut_indexs:
    #         #new_bee[i] = random.uniform(lb, ub)
    #         new_bee[i] = bee[i] * random.random()


    return (new_bee)
    #
    # else:
    #     mut_indexs = random.sample(range(dimension), num)
    #     new_bee = bee.copy()
    #     for i in mut_indexs:
    #         new_bee[i] = bee[i] + random.uniform(lb, ub) * random.random()
    #
    #     return (new_bee)



def send_onlookers(pop, fitness, func, lb, ub, dimension):
    #w_indexs = np.argwhere(fitness > fitness.mean())
    #b_indexs = np.argwhere(fitness <= fitness.mean())
    new_pop = pop.copy()
    pop_size = int(len(fitness))
    meio_size = int(pop_size/2)
    # terco_size = int(pop_size/3)
    quarto_size = int(pop_size/4)

    arr_idx = np.arange(pop_size).reshape(pop_size,1)
    merge = np.concatenate((arr_idx,fitness),axis=1)
    merge = merge[np.argsort(merge[:, 1])][::-1]


    for i in range(quarto_size, meio_size):
        #num_rand = random.randrange(terco_size*2, pop_size)
        if random.randint(0, 20) == 0:  # 5%
            num_rand = random.randrange(meio_size, pop_size)
            new_pop[int(merge[i,0])] = pop[int(merge[num_rand,0])]
            #trials[i] = 0


    for i in range(0,quarto_size): # CRUZAMENTO
    #for i in range(meio_size): # CRUZAMENTO
        if random.randint(0,5) == 0: # 5%
            num_rand = random.randrange(meio_size, pop_size)
            new_pop[int(merge[i, 0])] = crossover_bee(new_pop[int(merge[i, 0])], pop[int(merge[num_rand, 0])], lb, ub, dimension)
            #trials[i] = 0

    new_fit = np.apply_along_axis(func, 1, new_pop).reshape(pop_size,1)
    return (new_pop, new_fit)



def send_scout(pop, lb, ub, dimension, fitness, func, trials, limit):
    tired = np.argwhere(trials > limit).reshape(-1).tolist()

    if len(tired) > 0:
        # print ("nego cansado: ", len(tired))
        # print (tired[0])
        # print (tired)
        # print(len(tired[0]))
        num = max(int(dimension/4),1)
        mut_indexs = random.sample(range(dimension), num)
        #for i in range(len(tired)):
        for i in tired:

            if i != np.argmin(fitness): # NAUM MEXA NA MELHOR ABELHA (ELITISMO)
                bee = np.random.uniform(low=lb, high=ub, size=dimension)

                pop[i][mut_indexs] = bee[mut_indexs]
                #pop[i] = bee

                if func(pop[i]) < np.max(fitness):
                    trials[i] = 0
                # else:
                #     trials[i] = 10 # pra voltar aqui mais cedo
                #print ("tá coisando")
            #print("depois: ", pop[i])
            # else:
            #     print ("nao muta")
            #     # print (i)
            #     # print (np.argmin(fitness))
            #     # #print (len(tired[0]))
            #     # print (trials)


    return (pop, trials)


def crossover_bee(*args):
    bee_worst = args[0]
    bee_best = args[1]
    # lb = args[2]
    # ub = args[3]
    dimension = args[4]

    #bee_rand = np.random.uniform(low=lb, high=ub, size=dimension)
    random_indexs = random.sample(range(dimension), max(int(dimension / 2),1))

    # for i in random_indexs:
    #     bee_worst[i] = bee_best[i]
    if random.randint(0,40) == 0: # 2.5% de chance
        for i in random_indexs:
            bee_worst[i] = (bee_best[i] + bee_worst[i]) / 2
    else:
        for i in random_indexs:
            bee_worst[i] = bee_best[i]
    return bee_worst


def optimize_custom(func, lb, ub, pop_size, dimension, interactions, limit):

    pop, fitness = init_population(func, lb, ub, pop_size, dimension)
    trials = np.zeros((pop_size,), dtype=int)
    testes = 0
    #print (trials)
    new_pop = np.full((pop_size, dimension), np.nan)
    for i in range(interactions):
        #print (pop[0])
        # send the employed bees
        #new_pop, new_fitness, trials = send_employee(func, lb, ub, pop, fitness, dimension, trials, limit)
        new_pop, new_fitness = send_employee(func, lb, ub, pop, fitness, dimension, trials, limit)

        # send the onlooker bees
        new_pop, new_fitness = send_onlookers(new_pop, new_fitness, func, lb, ub, dimension)

        # send scout bees
        new_pop, trials = send_scout(new_pop, lb, ub, dimension, new_fitness, func, trials, limit)

        pop = new_pop
        fitness = np.apply_along_axis(func, 1, pop).reshape(pop_size,1)
        trials = trials + 1


    #     if testes == 100: #or testes == 200 or testes == 300 or testes == 400:
    #         print ("min: ", np.min(fitness))
    #         #print ("max: ", np.max(fitness))
    #         #print ("media: ", np.mean(fitness))
    #         #print (pop[np.argmin(fitness)])
    #
    #
    #         testes = 0
    #     testes += 1
    # print("\n")

    best_result = np.argmin(fitness)
    return (pop[best_result], np.min(fitness))
