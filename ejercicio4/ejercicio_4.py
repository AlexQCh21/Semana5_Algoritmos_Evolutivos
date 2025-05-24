import pandas as pd
import random

random.seed(42)
df = pd.read_csv("ejercicio4/Projects.csv", sep=";")
proyectos = df.to_dict("records")

def generar_solucion_inicial(proyectos): 
    #bitstring aleatorio como solución inicial
    solucion_inicial = [random.choice([0,1]) for _ in proyectos]
    return solucion_inicial

def evaluar_fitness(solucion, proyectos, presupuesto):
    costo_total = 0
    beneficio_total = 0
    
    for bit, proyecto in zip(solucion, proyectos):
        if bit == 1:
            costo_total += proyecto.get('Cost_Soles', 0)
            beneficio_total += proyecto.get('Benefit_Soles', 0)

    
    if costo_total > presupuesto:
        return -float('inf')
        
    return beneficio_total

def generar_vecino(solucion):
    vecino = solucion.copy()
    posicion = random.randint(0, len(vecino) - 1)
    vecino[posicion] = 1 - vecino[posicion]
    return vecino

def hill_climbing(proyectos, presupuesto_maximo, max_iter=1000):
    # Paso 1: solución inicial aleatoria
    solucion_actual = generar_solucion_inicial(proyectos)
    fitness_actual  = evaluar_fitness(solucion_actual, proyectos, presupuesto_maximo)
    
    for _ in range(max_iter):
        # Paso 3: generar vecino (cambiar un bit)
        vecino = generar_vecino(solucion_actual)
        fitness_vecino = evaluar_fitness(vecino, proyectos, presupuesto_maximo)
        
        # Paso 5 y 6: si vecino es mejor, avanzar, sino terminar
        if fitness_vecino > fitness_actual:
            solucion_actual = vecino
            fitness_actual = fitness_vecino
        else:
            break  # no mejoró, termina
    
    return solucion_actual, fitness_actual

def hill_climbing_multiple_intentos(proyectos, presupuesto_maximo, intentos=10):
    mejor_solucion = None
    mejor_fitness = -float('inf')

    for _ in range(intentos):
        solucion, fitness = hill_climbing(proyectos, presupuesto_maximo)
        if fitness > mejor_fitness:
            mejor_fitness = fitness
            mejor_solucion = solucion

    return mejor_solucion, mejor_fitness


# solucion, fitness = hill_climbing(proyectos, 10000, max_iter=1000)
solucion, fitness = hill_climbing_multiple_intentos(proyectos, 10000, intentos=20)


proyectos_seleccionados = [p['ProjectID'] for p, b in zip(proyectos, solucion) if b == 1]
costo_total = sum(p['Cost_Soles'] for p, b in zip(proyectos, solucion) if b == 1)

print(f"Proyectos seleccionados: {proyectos_seleccionados}")
print(f"Beneficio total: S/ {fitness}")
print(f"Costo total: S/ {costo_total}")

