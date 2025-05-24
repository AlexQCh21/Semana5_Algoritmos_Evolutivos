import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from deap import base, creator, tools

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. Semillas para reproducibilidad
random.seed(42)
np.random.seed(42)

# 2. Cargar dataset
df = pd.read_csv("ejercicio10/Enrollments.csv", sep=';')

# Features y etiquetas
X = df[['Credits', 'Prev_GPA', 'Extracurricular_hours']].values
y = df['Category'].values

# Codificar categorías (Alta=0, Media=1, Baja=2)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Dividir train/val 70/30
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.3, random_state=42)

# 3. Definir función para crear la red con genotipo
def build_mlp(individual):
    # individual = [num_layers, neurons1, neurons2, neurons3, lr]
    num_layers = individual[0]
    neurons = individual[1:4][:num_layers]
    lr = individual[4]

    # Convertir a int la neuronas, capa y asegurar rango
    num_layers = int(np.clip(num_layers, 1, 3))
    neurons = [int(np.clip(n, 1, 10)) for n in neurons]
    lr = float(np.clip(lr, 0.001, 0.1))

    # Definir capas ocultas
    hidden_layers = tuple(neurons)

    # Crear modelo MLP
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=lr,
                          max_iter=20, random_state=42)
    return model

# 4. Fitness: accuracy en validación
def fitness(individual):
    model = build_mlp(individual)
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return (acc,)

# 5. Generar vecinos para hill climbing (pequeños ajustes)
def get_neighbors(ind):
    neighbors = []
    step_int = 1
    step_float = 0.01

    # Variar num_layers (1-3)
    for delta in [-step_int, step_int]:
        new_ind = ind[:]
        new_ind[0] = int(np.clip(new_ind[0] + delta, 1, 3))
        neighbors.append(new_ind)

    # Variar neuronas en cada capa (1-10)
    for i in range(1,4):
        for delta in [-step_int, step_int]:
            new_ind = ind[:]
            new_ind[i] = int(np.clip(new_ind[i] + delta, 1, 10))
            neighbors.append(new_ind)

    # Variar learning rate (0.001-0.1)
    for delta in [-step_float, step_float]:
        new_ind = ind[:]
        new_ind[4] = float(np.clip(new_ind[4] + delta, 0.001, 0.1))
        neighbors.append(new_ind)

    return neighbors

def hill_climbing_local(individual):
    current = individual[:]
    current_fit = fitness(current)[0]
    neighbors = get_neighbors(current)
    for neighbor in neighbors:
        f = fitness(neighbor)[0]
        if f > current_fit:
            return neighbor
    return current

# 6. DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Inicializar genes
toolbox.register("num_layers", random.randint, 1, 3)
toolbox.register("neurons", random.randint, 1, 10)
toolbox.register("learning_rate", random.uniform, 0.001, 0.1)

def init_individual():
    return creator.Individual([
        toolbox.num_layers(),
        toolbox.neurons(),
        toolbox.neurons(),
        toolbox.neurons(),
        toolbox.learning_rate()
    ])

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)

# Cruce: uniform
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# Mutación mejorada: permite paso 1 o 2 en neuronas (sin cambiar num_layers)
def mutate_individual(individual):
    idx = random.randint(0, 4)
    if idx == 0:  # num_layers fijo (no cambia)
        pass
    elif 1 <= idx <= 3:  # neuronas con paso 1 o 2
        step = random.choice([1, 2])
        direction = random.choice([-1, 1])
        individual[idx] = int(np.clip(individual[idx] + direction * step, 1, 10))
    else:  # learning_rate pequeño cambio
        individual[4] = float(np.clip(individual[4] + random.uniform(-0.01, 0.01), 0.001, 0.1))
    return (individual,)

toolbox.register("mutate", mutate_individual)

def mutate_hill(individual):
    mutant, = toolbox.mutate(individual)
    improved = hill_climbing_local(mutant)
    individual[:] = improved
    return (individual,)

toolbox.register("mutate_hill", mutate_hill)
toolbox.register("select", tools.selTournament, tournsize=3)

# 7. Algoritmo evolutivo
def main():
    NGEN = 30
    POPSIZE = 30

    pop = toolbox.population(n=POPSIZE)
    stats = []
    best_accs = []

    # Evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Cruce
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutación + hill climbing local
        for mutant in offspring:
            if random.random() < 0.3:
                toolbox.mutate_hill(mutant)
                del mutant.fitness.values

        # Evaluar inválidos
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalids)
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        best_fit = max(fits)
        mean_fit = np.mean(fits)
        stats.append((gen, best_fit, mean_fit))
        best_accs.append(tools.selBest(pop, 1)[0])

        print(f"Gen {gen} - Mejor accuracy: {best_fit:.4f} | Promedio: {mean_fit:.4f}")

    best_ind = tools.selBest(pop, 1)[0]
    print("\nMejor individuo:", best_ind)
    print("Accuracy validación:", best_ind.fitness.values[0])

    # Mostrar arquitectura
    print(f"Arquitectura: {int(best_ind[0])} capas, neuronas: {best_ind[1:4][:int(best_ind[0])]}, LR: {best_ind[4]:.4f}")

    # Gráfica
    gens, bests, means = zip(*stats)
    plt.plot(gens, bests, label="Mejor accuracy")
    plt.plot(gens, means, label="Promedio accuracy")
    plt.xlabel("Generación")
    plt.ylabel("Accuracy")
    plt.title("Evolución de accuracy en validación")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
