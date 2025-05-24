import pandas as pd
import numpy as np
import random

# Controlar aleatoriedad para reproducibilidad
np.random.seed(42)
random.seed(42)

# Cargar dataset
df = pd.read_csv("ejercicio7/Students.csv", sep=";")

# Codificar Skills en variables numéricas one-hot
skills_dummies = pd.get_dummies(df["Skill"])
# Concatenar para tener matriz skills numérica
skills_matrix = skills_dummies.values

# Extraer GPA en array numpy
gpas = df["GPA"].values

NUM_ALUMNOS = len(df)
NUM_EQUIPOS = 5
ALUMNOS_POR_EQUIPO = 4

def fitness(solucion):
    total_varianza_gpa = 0
    penalizacion_skills = 0

    for equipo in range(NUM_EQUIPOS):
        indices = [i for i, e in enumerate(solucion) if e == equipo]
        if len(indices) == 0:
            continue
        gpa_equipo = gpas[indices]
        skills_equipo = skills_matrix[indices]

        varianza_gpa = np.var(gpa_equipo)
        total_varianza_gpa += varianza_gpa

        # Penalización: suma de desviaciones estándar por skill
        penalizacion_skills += np.sum(np.std(skills_equipo, axis=0))

    return total_varianza_gpa + penalizacion_skills

def get_neighbors(solucion):
    vecinos = []
    for _ in range(10):
        vecino = solucion.copy()
        a1, a2 = random.sample(range(NUM_ALUMNOS), 2)
        if vecino[a1] != vecino[a2]:
            vecino[a1], vecino[a2] = vecino[a2], vecino[a1]
            vecinos.append(vecino)
    return vecinos

def hill_climbing(max_iter=1000):
    solucion_actual = [i % NUM_EQUIPOS for i in range(NUM_ALUMNOS)]
    random.shuffle(solucion_actual)

    fitness_actual = fitness(solucion_actual)
    historial = [fitness_actual]

    for _ in range(max_iter):
        vecinos = get_neighbors(solucion_actual)
        mejor_vecino = None
        mejor_fitness = float('inf')

        for vecino in vecinos:
            f = fitness(vecino)
            if f < mejor_fitness:
                mejor_fitness = f
                mejor_vecino = vecino

        if mejor_fitness >= fitness_actual:
            break

        solucion_actual = mejor_vecino
        fitness_actual = mejor_fitness
        historial.append(fitness_actual)

    return solucion_actual, fitness_actual, historial

def imprimir_resultado(solucion):
    for equipo in range(NUM_EQUIPOS):
        indices = [i for i, e in enumerate(solucion) if e == equipo]
        print(f"Equipo {equipo+1}:")
        print(df.loc[indices, ["StudentID", "GPA", "Skill"]])
        print()

if __name__ == "__main__":
    sol, fit, hist = hill_climbing()
    imprimir_resultado(sol)
    print(f"Fitness final: {fit:.4f}")
    print(f"Historial últimos 10 fitness: {hist[-10:]}")
