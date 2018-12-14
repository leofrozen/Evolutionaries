import numpy as np
from scipy.optimize import rosen
#from yabox.problems import
# import yabox.problems
# import yabox.problems.Schwefel as schwefel
# import yabox.problems.Levy as levy
# import yabox.problems.Griewank as griewank

from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

from single_objective.optimization_functions import levy, griewank, dixonprice


#yabox.problems.

# links:
# http://cleverowl.uk/2015/07/01/using-one-way-anova-and-tukeys-test-to-compare-data-sets/
#


from single_objective import CE, t_test, PE as ep, ed_python_alheio


def rosenbrock(x):
    # teste
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


#func = rosenbrock

all_functions = [rosen, levy, griewank, dixonprice]
#all_functions = [rosen]

dimension = 30
pop_size = 50
lb = -5
ub = 5

it = 9000
num_execs = 30


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
    best_ep = [0] * num_execs

    print ('-----------------------------------------------------')
    print ("---------------- FUNCTION ", func.__name__)
    print('-----------------------------------------------------')

    print ("-------GA----------")
    for i in range(num_execs):
        #print ("EXECUCAO NUM: ", i+1)
        result[i], best_ga[i] = CE.ga(rosen, lb, ub, pop_size, dimension, it, prob_cross=prob_cross, prob_muta=prob_muta, select_type=2, t_size=6, elitism=True)


    print("BEST GA: ")
    print (best_ga)

    print("\n\n")


    print ("-------DE----------")
    for i in range(num_execs):
       best_ed[i] = (ed_python_alheio.de(rosen, bounds, mut=fact, crossp=prob_cross, popsize=pop_size, its=it))

    print("BEST DE: ")
    print (best_ed)
    print("\n\n")


    print ("-------PSO----------")
    for i in range(num_execs):
        #print ("Execucao numero ", i+1)
        result[i], best_pso[i] = PSO.optimize_custom(func=func, lb=lb, ub=ub, v_max=vmin, v_min=vmin, swarm_size=pop_size, dimension=dimension, iterations=it, w_min=wmin, w_max=wmax, c1=c1, c2=c2)

    print("BEST PSO: ")
    print (best_pso)
    print("\n\n")

    print("-------EP----------")
    for i in range(num_execs):
        # print ("Execucao numero ", i+1)
        result[i], best_ep[i], std_dev = ep.optimize_custom(func=func, lb=lb, ub=ub, pop_size=pop_size, dimension=dimension, iterations=it, q=5)

    print("BEST EP: ")
    print(best_ep)
    print("\n\n")

    print("-------ABC----------")
    for i in range(num_execs):
        # print ("Execucao numero ", i+1)
        result[i], best_abc[i] = ABC.optimize_custom(func=func, lb=lb, ub=ub, pop_size=pop_size, dimension=dimension, interactions=it, limit=20)

    print("BEST ABC: ")
    print(best_abc)
    print("\n\n")




    print ("Teste T: GA vs ED\n")
    t_test.t_test(np.array(best_ga), np.array(best_ed), num_execs)
    print ("Teste T: GA vs PSO\n")
    t_test.t_test(np.array(best_ga), np.array(best_pso), num_execs)
    print ("Teste T: GA vs EP\n")
    t_test.t_test(np.array(best_ga), np.array(best_ep), num_execs)
    print("Teste T: GA vs ABC\n")
    t_test.t_test(np.array(best_ga), np.array(best_abc), num_execs)


    print ("Teste T: ED vs PSO\n")
    t_test.t_test(np.array(best_ed), np.array(best_pso), num_execs)
    print ("Teste T: ED vs EP\n")
    t_test.t_test(np.array(best_ed), np.array(best_ep), num_execs)
    print ("Teste T: ED vs ABC\n")
    t_test.t_test(np.array(best_ed), np.array(best_abc), num_execs)

    print("Teste T: PSO vs EP\n")
    t_test.t_test(np.array(best_pso), np.array(best_ep), num_execs)
    print("Teste T: PSO vs ABC\n")
    t_test.t_test(np.array(best_pso), np.array(best_abc), num_execs)

    print("Teste T: EP vs ABC\n")
    t_test.t_test(np.array(best_ep), np.array(best_abc), num_execs)

    print ('----------------------------FIM DA FUNCAO ', func.__name__)
    print ("\n\n\n")

    #
    #  http://cleverowl.uk/2015/07/01/using-one-way-anova-and-tukeys-test-to-compare-data-sets/
    #

    ##
    ##   ANOVA
    ##

    # Prepara os dados

    # arrayegua
    arrayegua = []
    for i in range(num_execs): arrayegua.append(("GA", best_ga[i]))
    for i in range(num_execs): arrayegua.append(("ED", best_ed[i]))
    for i in range(num_execs): arrayegua.append(("PSO", best_pso[i]))
    for i in range(num_execs): arrayegua.append(("EP", best_ep[i]))
    for i in range(num_execs): arrayegua.append(("ABC", best_abc[i]))

    # data_arr
    data_arr = np.rec.array(arrayegua, dtype=[('Algoritmo', '|U5'), ('Fitness', float)])

    ###

    print("Teste ANOVA: GA vs ED\n")
    f, p = stats.f_oneway(np.array(best_ga), np.array(best_ed))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: GA vs PSO\n")
    f, p = stats.f_oneway(np.array(best_ga), np.array(best_pso))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: GA vs EP\n")
    f, p = stats.f_oneway(np.array(best_ga), np.array(best_ep))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: GA vs ABC\n")
    f, p = stats.f_oneway(np.array(best_ga), np.array(best_abc))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: ED vs PSO\n")
    f, p = stats.f_oneway(np.array(best_ed), np.array(best_pso))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: ED vs EP\n")
    f, p = stats.f_oneway(np.array(best_ed), np.array(best_ep))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: ED vs ABC\n")
    f, p = stats.f_oneway(np.array(best_ed), np.array(best_abc))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: PSO vs EP\n")
    f, p = stats.f_oneway(np.array(best_pso), np.array(best_ep))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: PSO vs ABC\n")
    f, p = stats.f_oneway(np.array(best_pso), np.array(best_abc))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: EP vs ABC\n")
    f, p = stats.f_oneway(np.array(best_ep), np.array(best_abc))
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    print("Teste ANOVA: COMPLETO \n")
    f, p = stats.f_oneway(data_arr[data_arr['Algoritmo'] == 'GA'].Fitness,
                          data_arr[data_arr['Algoritmo'] == 'ED'].Fitness,
                          data_arr[data_arr['Algoritmo'] == 'PSO'].Fitness,
                          data_arr[data_arr['Algoritmo'] == 'EP'].Fitness,
                          data_arr[data_arr['Algoritmo'] == 'ABC'].Fitness)
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')

    ##
    ##   TUKEY
    ##
    print ('\n\n\n#### TESTE TUKEY ####')

    # arrayegua = []
    # for i in range(num_execs): arrayegua.append(("GA", best_ga[i]))
    # for i in range(num_execs): arrayegua.append(("ED", best_ed[i]))
    # for i in range(num_execs): arrayegua.append(("PSO", best_pso[i]))
    # for i in range(num_execs): arrayegua.append(("EP", best_ep[i]))
    # for i in range(num_execs): arrayegua.append(("ABC", best_abc[i]))
    #
    # data_arr = np.rec.array(arrayegua, dtype=[('Algoritmo', '|U5'), ('Fitness', float)])

    mc = MultiComparison(data_arr['Fitness'], data_arr['Algoritmo'])
    turkey_result = mc.tukeyhsd()

    print(turkey_result)
    print(mc.groupsunique)

##########################33

# bounds = [(-5,5)]*30
# result = differential_evolution(rosen, bounds)
# print (result.x)
# print("\n")
# print (result.fun)

#########################3
