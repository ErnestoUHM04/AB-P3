# Cornejo Morales Paola
# Hernández Martínez Ernesto Ulises

import random
import numpy as np

# Realizar la minimización de la función
#   x**2 + y**2 + [25*(sin(x) + sin(y))]
def fitness(x, y):
    return x**2 + y**2 + 25*[np.sin(x) + np.sin(y)]
# en el intervalo de valores (-5,5) para (x, y) usando PSO.

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


