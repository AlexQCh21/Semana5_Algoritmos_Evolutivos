import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from deap import base, creator, tools

# --- Configuración ---
random.seed(42)
np.random.seed(42)

# Cargar dataset
df = pd.read_csv("ejercicio9/Emails.csv", sep=';')
features = df.iloc[:, :-1].values  # columnas Feature1..Feature5
labels = df.iloc[:, -1].values     # columna Spam

# Dividir dataset en train y validación (70/30)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.3, random_state=42)

# --- Funciones ---
def classify(individual, X):
    # individual[0] = umbral
    # individual[1:6] = pesos para 5 features
    weights = np.array(individual[1:])
    scores = X @ weights  # producto punto
    return (scores >= individual[0]).astype(int)

def fitness(individual):
    y_pred = classify(individual, X_val)
    score = f1_score(y_val, y_pred)
    return (score,)

def get_neighbors(ind):
    neighbors = []
    step = 0.05
    for i in range(len(ind)):
        for delta in [-step, step]:
            neighbor = ind[:]
            neighbor[i] += delta
            # Limitar valores a [0,1]
            neighbor[i] = max(0.0, min(1.0, neighbor[i]))
            neighbors.append(neighbor)
    return neighbors

def hill_climbing_local(individual):
    current = individual[:]
    current_fitness = fitness(current)[0]
    neighbors = get_neighbors(current)
    for neighbor in neighbors:
        f = fitness(neighbor)[0]
        if f > current_fitness:
            return neighbor
    return current

# --- DEAP setup ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Genotipo: 6 floats [0,1]
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

def mut_hill(individual):
    # Mutar
    mutant, = toolbox.mutate(individual)
    # Limitar a [0,1]
    for i in range(len(mutant)):
        mutant[i] = max(0.0, min(1.0, mutant[i]))
    # Hill climbing local
    improved = hill_climbing_local(mutant)
    individual[:] = improved
    return (individual,)

toolbox.register("mutate_hill", mut_hill)

# --- Algoritmo Evolutivo ---
def main():
    pop = toolbox.population(n=50)
    NGEN = 40
    stats = []
    best_individuals = []

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Cruzamiento
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutación + hill climbing local
        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate_hill(mutant)
                del mutant.fitness.values

        # Recalcular fitness solo de los inválidos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Nueva población
        pop[:] = offspring

        # Guardar estadísticas
        fits = [ind.fitness.values[0] for ind in pop]
        best = max(fits)
        mean = np.mean(fits)
        stats.append((gen, best, mean))
        best_individuals.append(tools.selBest(pop, 1)[0])

        print(f"Gen {gen}: Best F1 = {best:.4f}, Mean F1 = {mean:.4f}")

    # Mejor individuo final
    best_overall = tools.selBest(pop, 1)[0]
    print("\nMejor individuo:", best_overall)
    print("Con F1-score =", best_overall.fitness.values[0])

    # Gráfica de convergencia
    gens, bests, means = zip(*stats)
    plt.plot(gens, bests, label="Best F1")
    plt.plot(gens, means, label="Mean F1")
    plt.xlabel("Generación")
    plt.ylabel("F1-score")
    plt.title("Convergencia de F1-score en evolución")
    plt.legend()
    plt.show()

    return best_overall

if __name__ == "__main__":
    best = main()
