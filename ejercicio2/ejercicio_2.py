import pandas as pd
import random

# Leer los datos
df = pd.read_csv("ejercicio2/MentorAvailability.csv",sep=";")
df.set_index("MentorID", inplace=True)

def calcular_choques(df, solucion):
    choques = 0
    for mentor_idx, slot_inicio in enumerate(solucion):
        disponibilidad = df.iloc[mentor_idx].values
        # Verifica si hay disponibilidad en ambos slots consecutivos
        if slot_inicio + 1 >= len(disponibilidad):
            choques += 2  # no se puede asignar, cuenta como choque doble
        else:
            if disponibilidad[slot_inicio] == 0:
                choques += 1
            if disponibilidad[slot_inicio + 1] == 0:
                choques += 1
    return choques

def generar_vecino(solucion, total_slots):
    vecino = solucion.copy()
    mentor_idx = random.randint(0, len(solucion) - 1)
    nuevo_slot = random.randint(0, total_slots - 2)  # -2 porque necesitamos dos horas consecutivas
    vecino[mentor_idx] = nuevo_slot
    return vecino

def busqueda_local(df, max_iter=1000):
    total_mentores = df.shape[0]
    total_slots = df.shape[1]
    
    # Inicialización aleatoria
    solucion_actual = [random.randint(0, total_slots - 2) for _ in range(total_mentores)]
    costo_actual = calcular_choques(df, solucion_actual)
    
    for _ in range(max_iter):
        vecino = generar_vecino(solucion_actual, total_slots)
        costo_vecino = calcular_choques(df, vecino)
        
        if costo_vecino < costo_actual:
            solucion_actual = vecino
            costo_actual = costo_vecino
        
        if costo_actual == 0:
            break
    
    return solucion_actual, costo_actual

solucion, choques = busqueda_local(df)
print("Horario final asignado:", solucion)
print("Total de choques:", choques)
