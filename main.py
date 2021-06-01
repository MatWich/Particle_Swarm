try:
    import random
    import math
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("cannot import all modules")

''' CZESC MODYFIKOWALNA '''

""" FUNKCJA CELU: https://en.wikipedia.org/wiki/Test_functions_for_optimization"""
def objective_function(X):
    """ X is a list of values of all x, y, z etc.
    Returns value of the function"""
    A = 10
    y = A * 2 + sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])
    # y = sum([x ** 2 for x in X])  # 2
    return y


bounds = [(-5.12, 5.12), (-5.12, 5.12)]  # upper and lower bounds of variables
# bounds = [(-float("inf"), float("inf")), (-float("inf"), float("inf"))]   # 2

nv = 2  # number of variables if 2 that means x, y
mm = -1  # if minimization problem, mm = -1; if maximization problem, mm = 1

particle_no = 50
iterations = 100
w = .75  # inertia constant
c1 = 1  # cognative constant
c2 = 2  # social constant

''' KONIEC CZESCI MODYFIKOWALNEJ '''

if mm == -1:
    initial_fitness = float("inf")  # for minimalization problem
elif mm == 1:
    initial_fitness = -float("inf")  # for maxymalization problem
else:
    raise Exception("You can only chooose 1 or -1")

# Visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()


class Particle:
    def __init__(self, bounds):
        """list of bounds for every variable in tuple"""
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity

    def evaluate(self, objective_function):
        """ update (fitness_)local_best_particle_position"""
        self.fitness_particle_position = objective_function(self.particle_position)

        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
        if mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best

    def update_velocity(self, global_best_particle_position):
        """ update velocity """
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        """ add velocity to the pos """
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]

            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]


""" ALGORITHM """

fitness_global_best_particle_position = initial_fitness
global_best_particle_position = []
swarm_particle = []
for i in range(particle_no):
    swarm_particle.append(Particle(bounds))
A = []

for i in range(iterations):
    for j in range(particle_no):
        swarm_particle[j].evaluate(objective_function)

        if mm == -1:
            if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                global_best_particle_position = list(swarm_particle[j].particle_position)
                fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)

        if mm == 1:
            if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                global_best_particle_position = list(swarm_particle[j].particle_position)
                fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)

    # update vel and pos of each particle
    for j in range(particle_no):
        swarm_particle[j].update_velocity(global_best_particle_position)
        swarm_particle[j].update_position(bounds)
    A.append(fitness_global_best_particle_position) # the best fitness

    # VISUALISATION
    ax.plot(A, color='g')
    fig.canvas.draw()

print(f"Optymalne rozwiazanie: {global_best_particle_position}")
print(f"Wartosc funkcji celu: {fitness_global_best_particle_position}")

plt.show()