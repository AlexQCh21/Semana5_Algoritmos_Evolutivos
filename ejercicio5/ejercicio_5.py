import pandas as pd
import random

df = pd.read_csv("ejercicio5/Tesistas.csv", sep=";")

# Convertir el dataframe en una estructura que permita trabajar facilmente con las disponibilidades de los tesistas
tesistas = []

for _, row in df.iterrows():
    tesista_id = row["TesistaID"]
    franjas_disponibles = [f"F{i+1}" for i in range(6) if row[f"F{i+1}"]==1]
    tesistas.append({
        "TesistaID":tesista_id,
        "Disponibles": franjas_disponibles
    })


def solucion_inicial(tesistas):
    solucion = []
    salas = [f"S{i+1}" for i in range(6)]
    for tesista in tesistas:
        if tesista["Disponibles"]:
            eleccion = random.choice(tesista["Disponibles"])
            eleccion_sala = random.choice(salas)
        else:
            eleccion = None
            eleccion_sala = None
        solucion.append({
            "TesistaID": tesista["TesistaID"],
            "FranjaElegida": eleccion,
            "Sala": eleccion_sala
        })
    return solucion

def evaluar_fitness(solucion):
    # Calcula:
    # - solapamientos (más de 1 tesista en misma sala y franja)
    # - huecos en salas (franjas intermedias libres)
    # - franjas continuas > 4 horas en sala
    # Devuelve un score o penalización total
    
    uso = {}
    for asignacion in solucion:
        sala = asignacion["Sala"]
        franja = asignacion["FranjaElegida"]
        if franja is None:
            continue
        clave = (sala, franja)
        uso.setdefault(clave, []).append(asignacion["TesistaID"])
    
    # Solapamientos
    solapamientos = 0
    for clave, tesistas_en_franja in uso.items():
        if len(tesistas_en_franja) > 1:
            solapamientos += len(tesistas_en_franja) - 1  # solo se penaliza el exceso

    # Franjas ocupadas por sala
    sala_franjas_ocupadas = {}
    for (sala, franja) in uso:
        sala_franjas_ocupadas.setdefault(sala, []).append(franja)

    # Ordenar franjas por sala
    sala_franjas_ocupadas_ordenada = {}
    for sala, franjas in sala_franjas_ocupadas.items():
        franjas_ordenadas = sorted(franjas, key=lambda x: int(x[1:]))  # F1 → 1
        sala_franjas_ocupadas_ordenada[sala] = franjas_ordenadas

    # Huecos en las franjas
    huecos = 0
    for sala, franjas in sala_franjas_ocupadas_ordenada.items():
        franjas_num = sorted([int(f[1:]) for f in franjas])
        for i in range(1, len(franjas_num)):
            salto = franjas_num[i] - franjas_num[i - 1]
            if salto > 1:
                huecos += (salto - 1)

    # Penalización por franjas continuas > 4
    penalizacion_cadena_larga = 0
    for sala, franjas in sala_franjas_ocupadas_ordenada.items():
        franjas_num = sorted([int(f[1:]) for f in franjas])
        max_consecutivas = 1
        consecutivas = 1
        for i in range(1, len(franjas_num)):
            if franjas_num[i] == franjas_num[i - 1] + 1:
                consecutivas += 1
                max_consecutivas = max(max_consecutivas, consecutivas)
            else:
                consecutivas = 1
        if max_consecutivas > 4:
            penalizacion_cadena_larga += (max_consecutivas - 4)

    # Calcular penalización total (puedes ajustar los pesos)
    penalizacion_total = (
        solapamientos * 10 +
        huecos * 2 +
        penalizacion_cadena_larga * 5
    )

    return penalizacion_total

def generar_vecino(solucion, tesistas):
    nuevo = solucion.copy()
    i = random.randint(0, len(nuevo) - 1)  # Seleccionar un tesista aleatorio
    
    tesista_id = nuevo[i]["TesistaID"]
    tesista_info = next(t for t in tesistas if t["TesistaID"] == tesista_id)
    
    if not tesista_info["Disponibles"]:
        return nuevo  # No tiene franjas disponibles, no se cambia

    # Cambiar a otra franja disponible (distinta si se puede)
    disponibles = tesista_info["Disponibles"]
    franja_actual = nuevo[i]["FranjaElegida"]
    posibles = [f for f in disponibles if f != franja_actual]
    nueva_franja = random.choice(posibles) if posibles else franja_actual

    # Cambiar a otra sala (distinta si se puede)
    salas = [f"S{i+1}" for i in range(6)]
    sala_actual = nuevo[i]["Sala"]
    posibles_salas = [s for s in salas if s != sala_actual]
    nueva_sala = random.choice(posibles_salas) if posibles_salas else sala_actual

    # Asignar nuevo valor
    nuevo[i] = {
        "TesistaID": tesista_id,
        "FranjaElegida": nueva_franja,
        "Sala": nueva_sala
    }
    return nuevo

def hill_climbing(tesistas, max_iter=1000):
    actual = solucion_inicial(tesistas)
    mejor_score = evaluar_fitness(actual)
    
    for _ in range(max_iter):
        vecino = generar_vecino(actual, tesistas)
        score_vecino = evaluar_fitness(vecino)
        
        if score_vecino < mejor_score:
            actual = vecino
            mejor_score = score_vecino
    
    return actual, mejor_score

        
if __name__ == "__main__":
    mejor_solucion, puntaje = hill_climbing(tesistas, max_iter=5000)
    print("Mejor solución encontrada:")
    for t in mejor_solucion:
        print(t)
    print("Penalización total:", puntaje)
