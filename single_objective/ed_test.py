from single_objective import CE
import numpy as np


def rosenbrock(x):
    # teste
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


lista = np.array([1,1,1,1,1])
#print (rosenbrock(lista))



dimension = 30
it = 1000
pop_size = 50
lb = -5
ub = 5
func = rosenbrock
num_execs = 3

result = [0]*num_execs
best = [0]*num_execs
prob_cross = 0.6
fact= 0.8
strategy = 2


print ("###  PARAMETROS: ")
print ("tamanho da populacao: ", pop_size)
print ("dimensao: ", dimension)
print ("iteracoes do GA: ",it)
print ("limite inferior: ", lb)
print ("limite superior: ", ub)
print ("probabilidade crossover: ", prob_cross)
print ("factor: ", fact)
print ("estrategia: ", strategy)
#print ("tipo de selecao: torneio")

for i in range(num_execs):
    print ("ED - EXECUCAO NUM: ", i+1)
    #result[i], best[i] = CE.ga(rosenbrock,lb,ub,pop_size,dimension,it,prob_cross=prob_cross,prob_muta=prob_muta,select_type=2,t_size=6,elitism=True)
    result[i], best[i] = CE.ed(rosenbrock, lb, ub, pop_size, dimension, it, fact, prob_cross, strategy)


print("BEST: ")
print (best)


print ("\n\n\nRESULT:")
for i in result:
    print (i)