import random
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# --- Control de aleatoriedad ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import pandas as pd
import numpy as np

# Cargar CSV con separador ';'
df = pd.read_csv("ejercicio8/HousePrices.csv", sep=";")

# Verificar columnas (opcional)
print(df.columns)  # Debería mostrar Index(['Rooms', 'Area_m2', 'Price_Soles'], dtype='object')

# Seleccionar variables predictoras y objetivo
X = df[["Rooms", "Area_m2"]].values
y = df["Price_Soles"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# --- Función fitness: evaluar RMSE de Ridge con un alpha dado ---
def eval_ridge(individual):
    # individual es una lista con un solo valor alpha
    alpha = individual[0]
    # Evitar alphas negativos o cero (penalizar fuerte)
    if alpha <= 0:
        return 1e6,
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse,

# --- Setup DEAP ---

# 1. Crear clase Fitness y Individual (minimizar RMSE)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar RMSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 2. Definir atributos: alpha entre 0.001 y 10 (rango razonable para Ridge)
toolbox.register("attr_alpha", random.uniform, 0.001, 10.0)

# 3. Crear individuo y población
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_alpha, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 4. Registrar función de evaluación (fitness)
toolbox.register("evaluate", eval_ridge)

# 5. Registrar mutación gaussiana pequeña, sin cruce
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=1.0)

# 6. Selección greedy: siempre escoge el mejor vecino
def sel_best(individuals, k):
    sorted_ind = sorted(individuals, key=lambda ind: ind.fitness.values)
    return sorted_ind[:k]

toolbox.register("select", sel_best)

# --- Algoritmo hill climbing con población (modo "algorithms.eaSimple" sin cruce) ---
def main():
    POP_SIZE = 20
    NGEN = 50
    MUTPB = 1.0  # Siempre mutar
    CXPB = 0.0   # No cruza

    pop = toolbox.population(n=POP_SIZE)

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitness_history = []

    for gen in range(NGEN):
        # Selección de los mejores para reproducirse
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Mutación
        for mutant in offspring:
            toolbox.mutate(mutant)
            # Limitar alpha a rango válido tras mutación
            if mutant[0] < 0.001:
                mutant[0] = 0.001
            elif mutant[0] > 10.0:
                mutant[0] = 10.0
            mutant.fitness.values = toolbox.evaluate(mutant)

        # Reemplazo (greedy: solo si mejora)
        combined = pop + offspring
        combined = toolbox.select(combined, POP_SIZE)

        pop[:] = combined

        # Guardar mejor fitness actual
        best = tools.selBest(pop, 1)[0]
        fitness_history.append(best.fitness.values[0])
        print(f"Gen {gen+1}: Mejor RMSE = {best.fitness.values[0]:.4f}, alpha = {best[0]:.4f}")

    mejor_ind = tools.selBest(pop, 1)[0]
    return mejor_ind, fitness_history

if __name__ == "__main__":
    mejor_individuo, historial = main()

    print(f"\nAlpha óptimo encontrado: {mejor_individuo[0]:.4f}")
    print(f"RMSE óptimo: {mejor_individuo.fitness.values[0]:.4f}")

    # Graficar curva de convergencia
    plt.plot(historial)
    plt.title("Convergencia de RMSE en Hill Climbing con DEAP")
    plt.xlabel("Generación")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()
