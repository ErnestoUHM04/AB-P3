# Cornejo Morales Paola
# Hernández Martínez Ernesto Ulises

import random
import numpy as np

# Realizar la minimización de la función
#   x**2 + y**2 + [25*(sin(x) + sin(y))]
# en el intervalo de valores (-5,5) para (x, y) usando PSO.
MAX_LIMIT = 5
MIN_LIMIT = -5

# Versión de PSO: Global
# Número de partículas: 20
c_particulas = 20
# Iteraciones: 50
c_iteraciones = 50

# ====== CONSTANTES
# a = 0.8    (inercia)
a = 0.8
# b1 = 0.7  (aprendizaje local)
b1 = 0.7
# b2 = 1.2   (aprendizaje global)
b2 = 1.2

# En cada iteración imprimir:
# Posición de la partícula (Trayectoria)
# Velocidad
# pbest (mejor individuo de esa particula)
# gbest (mejor individuo OVERALL de entre todas las particulas)

def fitness(x, y):
    return (x**2 + y**2 + (25 * (np.sin(x) + np.sin(y))))

def create_generation():
    new_generation = []

    for i in range(c_particulas):
        # Individual is particle
        # Each particle has to have:    Position Vector (X,Y), and Velocity Vector (Vx, Vy)
        # We will use the Position Vector to get their fitness
        X = random.uniform(-5, 5)
        Y = random.uniform(-5, 5)
        fit = fitness(X, Y)
        # When we initialize Velocity Vector is always (0,0)
        individual = [(X,Y),fit,(0,0)]
        new_generation.append(individual)

    return new_generation

def print_generation(generation):
    for i in range(c_particulas):
        # To read the individual
        (pos, fit, vel) = generation[i]
        (x, y) = pos
        (vx, vy) = vel
        print("Particle", i, f"\tPosition: ({x:.4f}, {y:.4f})", f"\t | Velocity :({vx:.4f}, {vy:.4f})", f"\t | Fitness: {fit:.6f}")
    print("\n")

def print_gbest(gbest):
    (pos, fit, vel) = gbest
    (x, y) = pos
    (vx, vy) = vel
    print("Global best: ", f"\tPosition: ({x:.4f}, {y:.4f})", f"\t | Velocity :({vx:.4f}, {vy:.4f})", f"\t | Fitness: {fit:.6f}", "\n")

def get_pbest_gbest(generation, old_pbest_list = None, old_gbest = None):
    # There will we a pbest for every particle, but only will be one gbest
    pbest_list = []

    if old_gbest is None:
        gbest = None
    else:
        gbest = old_gbest

    for i in range(c_particulas):
        fit = generation[i][1]

        # Conditions for the initial generation
        if gbest is None:
            gbest = generation[i]

        if fit < gbest[1]:
            # Then we update the gbest
            gbest = generation[i]

        if old_pbest_list is None:
            pbest = generation[i]
            pbest_list.append(pbest)
            continue

        if fit < old_pbest_list[i][1]:
            # Then we update that pbest
            pbest = generation[i]
        else:
            pbest = old_pbest_list[i]

        pbest_list.append(pbest)

    return pbest_list, gbest

def update_generation(old_generation, pbest_list, gbest):
    new_generation = []
    for i in range(c_particulas):
        pbest = pbest_list[i]

        pos = old_generation[i][0]
        (X, Y) = pos

        vel = old_generation[i][2]
        (Vx, Vy) = vel

        nVx, nVy = update_velocity(X, Y, Vx, Vy, pbest, gbest)

        X += nVx

        if X > MAX_LIMIT:
            X = MAX_LIMIT # Round it up to 5
        if X < MIN_LIMIT:
            X = MIN_LIMIT # Round it up to -5

        Y += nVy

        if Y > MAX_LIMIT:
            Y = MAX_LIMIT
        if Y < MIN_LIMIT:
            Y = MIN_LIMIT

        fit = fitness(X, Y)

        individual = [(X,Y),fit,(nVx,nVy)]

        new_generation.append(individual)

    return new_generation

def update_velocity(X, Y, Vx, Vy, pbest, gbest):
    (pb_X, pb_Y) = pbest[0]
    (gb_X, gb_Y) = gbest[0]

    r1 = random.random()
    r2 = random.random()

    nVx = a * Vx + b1 * r1 * (pb_X - X) + b2 * r2 * (gb_X - X)
    nVy = a * Vy + b1 * r1 * (pb_Y - Y) + b2 * r2 * (gb_Y - Y)

    return nVx, nVy

num_gen = 0
generation = create_generation()
print("\tInitial generation")
#print_generation(generation)# <- THIS IS FOR DEBUGGING
# After each generation we need to save/update every pbest and gbest
pbest_list, gbest = get_pbest_gbest(generation)
print_gbest(gbest)
print("Particles best: \n")
print_generation(pbest_list)

while True:
    num_gen += 1
    generation = update_generation(generation, pbest_list, gbest)
    # Then we update the pbest list and the gbest
    pbest_list, gbest = get_pbest_gbest(generation, pbest_list, gbest)
    print("\tGeneration: ", num_gen)
    print_generation(generation)# <- THIS IS FOR DEBUGGING
    print_gbest(gbest)
    print("Particles best: \n")
    print_generation(pbest_list)# <- This is what the teacher asked us
    if num_gen >= c_iteraciones:
        break