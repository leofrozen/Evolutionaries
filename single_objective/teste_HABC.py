import numpy as np
from scipy.optimize import rosen
#from yabox.problems import
# import yabox.problems
# import yabox.problems.Schwefel as schwefel
# import yabox.problems.Levy as levy
# import yabox.problems.Griewank as griewank

from scipy import stats
from statsmodels.stats.multicomp import MultiComparison


from single_objective.optimization_functions import griewank, dixonprice, sphere, rastrigin, schwefel, ackley


#yabox.problems.

# links:
# http://cleverowl.uk/2015/07/01/using-one-way-anova-and-tukeys-test-to-compare-data-sets/
#


def rosenbrock(x):
    # teste
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


#func = rosenbrock

#all_functions = [rosen, levy, griewank, dixonprice]
#all_functions = [rosen]
all_functions = [rosen, griewank, dixonprice, sphere, rastrigin, schwefel, ackley]
#all_functions = [rosenbrock]

dimension = 30
#pop_size = 126
pop_size = 50
lb = -5
ub = 5

it = 10000
num_execs = 30
limite_abc = 100
limite_habc = 100

result = [0]*num_execs
best_ga = [0]*num_execs
best_ed = [0]*num_execs
best_pso = [0]*num_execs
best_ep = [0]*num_execs
best_abc = [0]*num_execs
prob_cross = 0.6
prob_muta=0.2
fact= 0.3
strategy = 2

bounds = [(-5,5)]*30


##  porqueiras do PSO

w = 0.7  # Inertia weight to prevent velocities becoming too large
wmin = 0.4
wmax = 0.9

c1 = 2  # Scaling co-efficient on the social component
c2 = 2  # Scaling co-efficient on the cognitive component

vmin = -100
vmax = 100
#############


for func in all_functions:

    result = [0] * num_execs
    best_ga = [0] * num_execs
    best_ed = [0] * num_execs
    best_pso = [0] * num_execs
    best_abc = [0] * num_execs
    best_habc = [0] * num_execs
    best_ep = [0] * num_execs

    print ('-----------------------------------------------------')
    print ("---------------- FUNCTION ", func.__name__)
    print('-----------------------------------------------------')

    if func.__name__ == "rosen":
        lb = -5.12
        ub = 5.12
        print ("Limites: {:f}, {:f}".format(lb,ub))
    elif func.__name__ == "griewank":
        lb = -600
        ub = 600
        print("Limites: {:f}, {:f}".format(lb, ub))
    elif func.__name__ == "dixonprice":
        lb = -10
        ub = 10
        print("Limites: {:f}, {:f}".format(lb, ub))
    elif func.__name__ == "sphere":
        lb = -5.12
        ub = 5.12
        print("Limites: {:f}, {:f}".format(lb, ub))
    elif func.__name__ == "rastrigin":
        lb = -15
        ub = 15
        print("Limites: {:f}, {:f}".format(lb, ub))
    elif func.__name__ == "schwefel":
        lb = -500
        ub = 500
        print("Limites: {:f}, {:f}".format(lb, ub))
    elif func.__name__ == "ackley":
        lb = -32.768
        ub = 32.768
        print("Limites: {:f}, {:f}".format(lb, ub))





    # print ("-------GA----------")
    # for i in range(num_execs):
    #     #print ("EXECUCAO NUM: ", i+1)
    #     result[i], best_ga[i] = CE.ga(rosen,lb,ub,pop_size,dimension,it,prob_cross=prob_cross,prob_muta=prob_muta,select_type=2,t_size=6,elitism=True)
    #
    #
    # print("BEST GA: ")
    # print (best_ga)
    #
    # print("\n\n")
    #
    #
    # print ("-------DE----------")
    # for i in range(num_execs):
    #    best_ed[i] = (ed_python_alheio.de(rosen,bounds, mut=fact, crossp=prob_cross, popsize=pop_size, its=it))
    #
    # print("BEST DE: ")
    # print (best_ed)
    # print("\n\n")

    #
    # print ("-------PSO----------")
    # for i in range(num_execs):
    #     #print ("Execucao numero ", i+1)
    #     result[i], best_pso[i] = pso.optimize_custom(func=func, lb=lb, ub=ub, v_max=vmin, v_min=vmin, swarm_size=pop_size, dimension=dimension, iterations=it, w_min=wmin, w_max=wmax, c1=c1, c2=c2)
    #
    # print("BEST PSO: ")
    # print (best_pso)
    # print("\n\n")
    #
    # print("-------EP----------")
    # for i in range(num_execs):
    #     # print ("Execucao numero ", i+1)
    #     result[i], best_ep[i], std_dev = ep.optimize_custom(func=func, lb=lb, ub=ub, pop_size=pop_size, dimension=dimension, iterations=it, q=5)
    #
    # print("BEST EP: ")
    # print(best_ep)
    # print("\n\n")
    #
    print("-------ABC----------")
    for i in range(num_execs):
        # print ("Execucao numero ", i+1)
        result[i], best_abc[i] = ABC.optimize_custom(func=func, lb=lb, ub=ub, pop_size=pop_size, dimension=dimension, interactions=it, limit=limite_abc)

    print("BEST ABC: ")
    print(best_abc)
    print("\n\n")

    print("-------HABC----------")
    for i in range(num_execs):
        # print ("Execucao numero ", i+1)
        result[i], best_habc[i] = HABC.optimize_custom(func=func, lb=lb, ub=ub, pop_size=pop_size, dimension=dimension, interactions=it, limit=limite_habc)

    print("BEST HABC: ")
    print(best_habc)
    print("\n\n")
    # print (result)

    ##
    ##   ANOVA
    ##

    # Prepara os dados

    # arrayegua
    arrayegua = []
    # for i in range(num_execs): arrayegua.append(("GA", best_ga[i]))
    # for i in range(num_execs): arrayegua.append(("ED", best_ed[i]))
    # for i in range(num_execs): arrayegua.append(("PSO", best_pso[i]))
    # for i in range(num_execs): arrayegua.append(("EP", best_ep[i]))
    for i in range(num_execs): arrayegua.append(("ABC", best_abc[i]))
    for i in range(num_execs): arrayegua.append(("HABC", best_habc[i]))

    # data_arr
    data_arr = np.rec.array(arrayegua, dtype=[('Algoritmo', '|U5'), ('Fitness', float)])

    print("Teste ANOVA: COMPLETO \n")
    f, p = stats.f_oneway(#data_arr[data_arr['Algoritmo'] == 'GA'].Fitness,
                          #data_arr[data_arr['Algoritmo'] == 'ED'].Fitness,
                          # data_arr[data_arr['Algoritmo'] == 'PSO'].Fitness,
                          # data_arr[data_arr['Algoritmo'] == 'EP'].Fitness,
                          data_arr[data_arr['Algoritmo'] == 'ABC'].Fitness,
                          data_arr[data_arr['Algoritmo'] == 'HABC'].Fitness)
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    ##
    ##   TUKEY
    ##
    print('\n\n\n#### TESTE TUKEY ####')

    mc = MultiComparison(data_arr['Fitness'], data_arr['Algoritmo'])
    turkey_result = mc.tukeyhsd()

    print(turkey_result)
    print(mc.groupsunique)



    ##  OUTROS DADOS
    print ("\n\n--DESVIO PADRAO--\n")
    print ("PSO: ", np.array(best_pso).std())
    print ("\n")
    print("EP: ", np.array(best_ep).std())
    print("\n")
    print("ABC: ", np.array(best_abc).std())
    print("\n")
    print("HABC: ", np.array(best_habc).std())
    print("\n")

    print("\n\n--MEDIAS--\n")
    print("PSO: ", np.array(best_pso).mean())
    print("\n")
    print("EP: ", np.array(best_ep).mean())
    print("\n")
    print("ABC: ", np.array(best_abc).mean())
    print("\n")
    print("HABC: ", np.array(best_habc).mean())
    print("\n")