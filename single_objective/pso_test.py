from single_objective.optimization_functions import dixonprice

def rosenbrock(x):
    # teste
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


w = 0.7  # Inertia weight to prevent velocities becoming too large
wmin = 0.4
wmax = 0.9

c1 = 2  # Scaling co-efficient on the social component
c2 = 2  # Scaling co-efficient on the cognitive component

vmin = -100
vmax = 100

swarmSize = 50
dimension = 30  # Size of the problem

it = 1000
lb = -5
ub = 5
func = dixonprice
num_execs = 5

result = [0]*num_execs
best_pso = [0]*num_execs


#S, F, V = pso.init_population(func,lb,ub,vmax,vmin,swarmSize,dimension)

for i in range(num_execs):
    print ("Execucao numero ", i+1)
    result[i], best_pso[i] = PSO.optimize_custom(func=func, lb=lb, ub=ub, v_max=vmin, v_min=vmin, swarm_size=swarmSize, dimension=dimension, iterations=it, w_min=wmin, w_max=wmax, c1=c1, c2=c2)

print("BEST PSO: ")
print (best_pso)
# for j in result:
#     print(j)

