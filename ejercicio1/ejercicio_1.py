import pandas as pd
import numpy as np

# Leer CSV con separador ';'
dataset = pd.read_csv("ejercicio1/grades.csv", sep=';')

print(dataset.head())

notas = dataset[['Parcial1', 'Parcial2', 'Parcial3']]

# Promedio individual
promedios_ind = notas.mean(axis=1)

# Promedio general
promedio_general = promedios_ind.mean()
print(f"Promedio general original: {promedio_general:.2f}")

# Porcentaje aprobados (promedio individual >= 11)
aprobados = (promedios_ind >= 11).mean() * 100
print(f"Porcentaje de aprobados original: {aprobados:.2f}%")


def hill_climbing(df, start=1, step=0.5):
    mejor_offset = start
    mejor_fitness = fitness(mejor_offset, df)

    while True:
        vecinos = [mejor_offset + step, mejor_offset - step]
        vecinos = [v for v in vecinos if -5 <= v <= 5]

        mejor_vecino = mejor_offset
        for v in vecinos:
            fit = fitness(v, df)
            if fit > mejor_fitness:
                mejor_fitness = fit
                mejor_vecino = v

        if mejor_vecino == mejor_offset:
            break
        else:
            mejor_offset = mejor_vecino

    return mejor_offset, mejor_fitness


def fitness(offset, df):
    notas_ajustadas = (df + offset).clip(lower=0)
    promedios = notas_ajustadas.mean(axis=1)
    porcentaje_aprobados = (promedios >= 11).mean() * 100
    promedio_general = promedios.mean()

    if promedio_general > 14:
        return -1
    return porcentaje_aprobados


resultado = hill_climbing(notas, start=0, step=0.5)
print("------------------------------RESULTADO------------------------------")
print(f"El mejor offset es {resultado[0]} y el mejor porcentaje de aprobados es {resultado[1]:.2f}%")