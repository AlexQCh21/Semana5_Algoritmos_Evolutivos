import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Para reproducibilidad
random.seed(42)
np.random.seed(42)

# Leer archivo con separador adecuado (puede ser ',' o ';' dependiendo del archivo)
df = pd.read_csv("ejercicio3/LabDistances.csv",sep=";", index_col=0)

# Convertimos a matriz numpy para los cálculos
dist_matrix = df.to_numpy()

# Nombres de los laboratorios (Lab1, Lab2, ...)
labs = df.index.tolist()

def calcular_distancia_total(ruta, matriz):
    distancia = 0
    for i in range(len(ruta) - 1):
        distancia += matriz[ruta[i]][ruta[i+1]]
    # Opcional: volver al punto inicial
    distancia += matriz[ruta[-1]][ruta[0]]
    return distancia

def generar_vecino(ruta):
    vecino = ruta.copy()
    i, j = random.sample(range(len(ruta)), 2)
    vecino[i], vecino[j] = vecino[j], vecino[i]
    return vecino

def hill_climbing(dist_matrix, iteraciones=1000):
    n = dist_matrix.shape[0]
    mejor_ruta = list(range(n))
    random.shuffle(mejor_ruta)
    mejor_distancia = calcular_distancia_total(mejor_ruta, dist_matrix)
    
    historial = [mejor_distancia]

    for _ in range(iteraciones):
        vecino = generar_vecino(mejor_ruta)
        distancia_vecino = calcular_distancia_total(vecino, dist_matrix)
        
        if distancia_vecino < mejor_distancia:
            mejor_ruta = vecino
            mejor_distancia = distancia_vecino
        
        historial.append(mejor_distancia)
    
    return mejor_ruta, mejor_distancia, historial


ruta_optima, distancia_optima, historial = hill_climbing(dist_matrix)

# Mostrar resultados
print("Ruta óptima (índices):", ruta_optima)
print("Ruta óptima (nombres):", [labs[i] for i in ruta_optima])
print("Distancia total:", distancia_optima)

# Graficar convergencia
plt.plot(historial)
plt.xlabel("Iteración")
plt.ylabel("Distancia total")
plt.title("Convergencia del Hill Climbing")
plt.grid(True)
plt.show()
