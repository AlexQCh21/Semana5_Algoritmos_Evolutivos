import pandas as pd
import random

# Carga del dataset
df = pd.read_csv("ejercicio6/ExamQuestions.csv", sep=";")

# Función de evaluación (fitness)
def evaluar(solucion, preguntas):
    total_dificultad = 0
    total_tiempo = 0
    for i, bit in enumerate(solucion):
        if bit == 1:
            total_dificultad += preguntas.iloc[i]["Difficulty"]
            total_tiempo += preguntas.iloc[i]["Time_min"]

    if total_tiempo > 90:
        return 9999  # penalización por exceder el tiempo
    if 180 <= total_dificultad <= 200:
        return 0  # solución ideal
    return abs(190 - total_dificultad)  # penaliza alejamiento del ideal

# Generar solución inicial aleatoria
def generar_solucion_inicial(n):
    return [random.randint(0, 1) for _ in range(n)]

# Generar vecino (bit flip)
def generar_vecino(solucion):
    vecino = solucion.copy()
    i = random.randint(0, len(vecino) - 1)
    vecino[i] = 1 - vecino[i]
    return vecino

# Algoritmo Hill Climbing
def hill_climbing(preguntas, max_iter=1000):
    actual = generar_solucion_inicial(len(preguntas))
    costo_actual = evaluar(actual, preguntas)

    for _ in range(max_iter):
        vecino = generar_vecino(actual)
        costo_vecino = evaluar(vecino, preguntas)
        if costo_vecino < costo_actual:
            actual = vecino
            costo_actual = costo_vecino

    return actual, costo_actual

# Ejecutar el algoritmo
mejor_solucion, mejor_costo = hill_climbing(df)

# Mostrar preguntas seleccionadas
preguntas_seleccionadas = df[[bool(b) for b in mejor_solucion]]
total_dificultad = preguntas_seleccionadas["Difficulty"].sum()
total_tiempo = preguntas_seleccionadas["Time_min"].sum()

# Resultados en pantalla
print("Mejor costo (penalización):", mejor_costo)
print("Total dificultad:", total_dificultad)
print("Total tiempo (min):", total_tiempo)
print("\nPreguntas seleccionadas:")
print(preguntas_seleccionadas.to_string(index=False))
