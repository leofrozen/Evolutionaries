import numpy as np
import random
import math



def init_population(func, lb, ub, pop_size, dimension):
    pop = np.random.uniform(low=lb, high=ub, size=(pop_size, dimension))
    fitness = np.apply_along_axis(func, 1, pop)
    #std_dev = np.random.uniform(low=lb, high=ub, size=(pop_size, dimension))
    std_dev = np.random.uniform(size=(pop_size, dimension))
    return (pop, fitness, std_dev)


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

def optimize_custom(func, lb, ub, pop_size, dimension,iterations,q):

    x, fit, std_dev = init_population(func, lb, ub, pop_size, dimension)
    tau = math.sqrt(2 * math.sqrt(pop_size)) ** -1
    taul = math.sqrt(2 * pop_size) ** -1

    testes = 0
    for i in range(iterations):
        # Cria filhos usando Cauchy mutation
        n1 = np.random.standard_cauchy((pop_size, dimension))
        xl = x + (std_dev * n1)

        #print (x.shape)
        #print (std_dev.shape)
        #print (n1.shape)


        # Verifica os limites de cada gene dos filhos
        xl = np.apply_along_axis(ajust_bounds, 1, xl, lb, ub, False)


        # Cria novos desvios
        #n1 = np.random.standard_normal(size=(pop_size,dimension))
        n1 = np.random.standard_normal(pop_size).reshape(pop_size,1)
        #print(n1.shape)
        #n_norm = np.random.randn()
        n_norm = np.random.standard_normal(1).reshape(1,1)
        #print (n_norm.shape)
        std_devl = std_dev * np.exp(taul * n_norm + tau * n1)
        fitl = np.apply_along_axis(func, 1, xl)

        # Faz a união entre pais e filhos
        tmp_pop = np.concatenate((xl, x), axis=0)
        #tmp_fit = np.union1d(fitl, fit).reshape(pop_size*2,1)
        tmp_fit = np.concatenate((fitl, fit),axis=0).reshape(pop_size*2,1)
        tmp_std = np.concatenate((std_devl, std_dev), axis=0)

        # Computa as vitórias de cada individuo
        win = np.full((pop_size*2), np.nan)
        for j in range(2 * pop_size):
            idx = random.sample(range(pop_size*2), q)
            win[j] = np.sum(tmp_fit[j] < tmp_fit[idx])

            #print (tmp_fit[j])
            #print (tmp_fit[idx])
        win = win.reshape(pop_size*2,1)

        #print (win[0])

        # Une as duas populações e ordena pelo numero de vitorias
        # Em seguida trunca-se pelo tamanho da população


        # print (win.shape)
        # print(tmp_fit.shape)
        # print(tmp_pop.shape)
        # print(tmp_std.shape)


        merge = np.concatenate((win, tmp_fit, tmp_pop, tmp_std),axis=1)
        #merge = merge[np.argsort(merge[:,0])]
        #merge = merge.argsort()[::-1][:,0]
        merge = merge[np.argsort(merge[:,0])][::-1]
        x = merge[0:pop_size, 2:(dimension + 2)]
        fit = merge[0:pop_size, 1]
        std_dev = merge[0:pop_size, (dimension + 2):(dimension * 2 + 3)]


        if testes == 100:
            #print (merge[0:5, 1])
            #print (merge[0:15,0])

            #print (np.min(fit))
            testes = 0
        testes += 1

    #return (x, fit, std_dev)
    return (x[fit.argmin()], np.min(fit), std_dev)


