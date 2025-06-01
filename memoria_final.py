"""
Memoria Final - Optimización de Rutas de Reparto con Búsquedas por Trayectorias
Implementación de Enfriamiento Simulado (ES) y Búsqueda Multiarranque Básica (BMB)
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from typing import List, Tuple, Dict, Any, Callable

# ---------------------------------
# Definición del problema de rutas
# ---------------------------------

class ProblemaRutasReparto:
    """Problema de optimización de rutas de reparto"""
    
    def __init__(self, num_clientes: int = 20, semilla: int = None):
        """
        Inicializar el problema de rutas de reparto
        
        Args:
            num_clientes: Número de clientes (puntos de entrega)
            semilla: Semilla para reproducibilidad
        """
        if semilla is not None:
            np.random.seed(semilla)
            random.seed(semilla)
            
        self.num_clientes = num_clientes
        # El punto 0 representa el almacén (origen y destino)
        self.num_puntos = num_clientes + 1
        
        # Generar coordenadas aleatorias para los puntos (almacén + clientes)
        self.coordenadas = np.random.rand(self.num_puntos, 2) * 100
        
        # Calculamos la matriz de distancias entre todos los puntos
        self.matriz_distancias = np.zeros((self.num_puntos, self.num_puntos))
        for i in range(self.num_puntos):
            for j in range(self.num_puntos):
                if i != j:
                    self.matriz_distancias[i, j] = self._calcular_distancia(i, j)
    
    def _calcular_distancia(self, punto1: int, punto2: int) -> float:
        """Calcular distancia euclidiana entre dos puntos"""
        return np.sqrt(np.sum((self.coordenadas[punto1] - self.coordenadas[punto2]) ** 2))
    
    def evaluar_ruta(self, ruta: List[int]) -> float:
        """
        Evaluar la distancia total de una ruta
        
        Args:
            ruta: Lista de índices de clientes (sin incluir el almacén)
            
        Returns:
            Distancia total de la ruta completa (incluyendo retorno al almacén)
        """
        # Convertir la ruta a una ruta completa que incluye el almacén al inicio y al final
        ruta_completa = [0] + ruta + [0]
        
        distancia_total = 0
        for i in range(len(ruta_completa) - 1):
            distancia_total += self.matriz_distancias[ruta_completa[i], ruta_completa[i + 1]]
        
        return distancia_total
    
    def generar_solucion_aleatoria(self) -> List[int]:
        """Generar una solución aleatoria (permutación de clientes)"""
        # Los clientes están enumerados del 1 al num_clientes
        solucion = list(range(1, self.num_puntos))
        random.shuffle(solucion)
        return solucion
    
    def graficar_ruta(self, ruta: List[int], titulo: str = "Ruta óptima"):
        """Visualizar una ruta en un plano 2D"""
        # Convertir la ruta a una ruta completa que incluye el almacén al inicio y al final
        ruta_completa = [0] + ruta + [0]
        
        plt.figure(figsize=(10, 8))
        
        # Dibujar los puntos
        plt.scatter(self.coordenadas[0, 0], self.coordenadas[0, 1], 
                   c='red', s=200, marker='*', label='Almacén')
        plt.scatter(self.coordenadas[1:, 0], self.coordenadas[1:, 1], 
                   c='blue', s=50, label='Clientes')
        
        # Dibujar las rutas
        for i in range(len(ruta_completa) - 1):
            plt.plot([self.coordenadas[ruta_completa[i], 0], self.coordenadas[ruta_completa[i + 1], 0]],
                    [self.coordenadas[ruta_completa[i], 1], self.coordenadas[ruta_completa[i + 1], 1]],
                    'k-', alpha=0.7)
            
        # Añadir etiquetas a los puntos
        for i in range(self.num_puntos):
            etiqueta = "Almacén" if i == 0 else f"Cliente {i}"
            plt.annotate(etiqueta, (self.coordenadas[i, 0] + 1, self.coordenadas[i, 1] + 1))
            
        plt.title(titulo)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
          # Guardar el gráfico
        filename = f"{titulo.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Gráfico guardado: {filename}")

# -----------------------------------
# Operadores para generar vecindarios
# -----------------------------------

def operador_intercambio(ruta: List[int]) -> List[int]:
    """
    Operador de intercambio (swap): intercambia dos clientes aleatorios
    
    Args:
        ruta: Ruta actual
        
    Returns:
        Nueva ruta con dos posiciones intercambiadas
    """
    nueva_ruta = ruta.copy()
    i, j = random.sample(range(len(ruta)), 2)
    nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
    return nueva_ruta

def operador_insercion(ruta: List[int]) -> List[int]:
    """
    Operador de inserción (shift): mueve un cliente a otra posición
    
    Args:
        ruta: Ruta actual
        
    Returns:
        Nueva ruta con un cliente movido
    """
    nueva_ruta = ruta.copy()
    i, j = random.sample(range(len(ruta)), 2)
    
    cliente = nueva_ruta.pop(i)
    nueva_ruta.insert(j, cliente)
    
    return nueva_ruta

# ---------------------------------
# Algoritmos de optimización
# ---------------------------------

def enfriamiento_simulado(
    problema: ProblemaRutasReparto,
    solucion_inicial: List[int] = None,
    ratio_t0: float = 0.3,
    tf: float = 0.001,
    max_evaluaciones: int = 100000,
    max_vecinos: int = None,
    ratio_max_exitos: float = 0.1,
    operador_vecindario: Callable = operador_intercambio
) -> Dict[str, Any]:
    """
    Algoritmo de Enfriamiento Simulado para optimización de rutas
    
    Args:
        problema: Instancia del problema
        solucion_inicial: Solución inicial (si es None, se genera aleatoriamente)
        ratio_t0: Porcentaje de empeoramiento para calcular T0
        tf: Temperatura final
        max_evaluaciones: Número máximo de evaluaciones
        max_vecinos: Máximo de vecinos por temperatura (si es None, se usa 10*n)
        ratio_max_exitos: Ratio máximo de éxitos por temperatura
        operador_vecindario: Función para generar vecinos
        
    Returns:
        Diccionario con resultados del algoritmo
    """
    # Inicialización
    if solucion_inicial is None:
        solucion_actual = problema.generar_solucion_aleatoria()
    else:
        solucion_actual = solucion_inicial.copy()
    
    coste_actual = problema.evaluar_ruta(solucion_actual)
    mejor_solucion = solucion_actual.copy()
    mejor_coste = coste_actual
    
    if max_vecinos is None:
        max_vecinos = 10 * len(solucion_actual)
    
    max_exitos = int(max_vecinos * ratio_max_exitos)
    
    # Calcular temperatura inicial T0 para aceptar soluciones un 30% peores
    # Usando la fórmula: exp(-delta/T0) = 0.5, donde delta = ratio_t0 * coste_inicial
    delta = coste_actual * ratio_t0
    t0 = -delta / math.log(0.5)
    
    # Esquema de enfriamiento de Cauchy modificado
    # Tk+1 = Tk / (1 + β*Tk), con β = (T0 - Tf) / (M*T0*Tf)
    m = max_evaluaciones // max_vecinos  # Número aproximado de temperaturas
    beta = (t0 - tf) / (m * t0 * tf)
    
    # Variables para seguimiento
    temperatura_actual = t0
    evaluaciones = 0
    historial = []
    
    print(f"Iniciando Enfriamiento Simulado:")
    print(f"  - Temperatura inicial: {t0:.2f}")
    print(f"  - Coste inicial: {coste_actual:.2f}")
    
    while evaluaciones < max_evaluaciones and temperatura_actual > tf:
        exitos = 0
        evaluaciones_locales = 0
        
        # Para cada nivel de temperatura, explorar el vecindario
        while exitos < max_exitos and evaluaciones_locales < max_vecinos:
            # Generar vecino
            vecino = operador_vecindario(solucion_actual)
            coste_vecino = problema.evaluar_ruta(vecino)
            evaluaciones += 1
            evaluaciones_locales += 1
            
            # Calcular delta y decidir si aceptar
            delta = coste_vecino - coste_actual
            
            if delta < 0 or random.random() < math.exp(-delta / temperatura_actual):
                solucion_actual = vecino
                coste_actual = coste_vecino
                exitos += 1
                
                # Actualizar mejor solución si corresponde
                if coste_actual < mejor_coste:
                    mejor_solucion = solucion_actual.copy()
                    mejor_coste = coste_actual
            
            # Registrar progreso
            if evaluaciones % 5000 == 0:
                historial.append((evaluaciones, mejor_coste))
                print(f"  Evaluaciones: {evaluaciones}, Mejor coste: {mejor_coste:.2f}, Temperatura: {temperatura_actual:.4f}")
        
        # Si no hay éxitos en esta temperatura, terminar
        if exitos == 0:
            print(f"  Sin éxitos en temperatura {temperatura_actual:.4f}. Terminando.")
            break
        
        # Enfriar temperatura usando esquema de Cauchy modificado
        temperatura_actual = temperatura_actual / (1 + beta * temperatura_actual)
    
    print(f"Enfriamiento Simulado finalizado:")
    print(f"  - Evaluaciones totales: {evaluaciones}")
    print(f"  - Mejor coste: {mejor_coste:.2f}")
    
    return {
        "mejor_solucion": mejor_solucion,
        "mejor_coste": mejor_coste,
        "evaluaciones": evaluaciones,
        "historial": historial,
        "temperatura_final": temperatura_actual
    }

def busqueda_local_basica(
    problema: ProblemaRutasReparto,
    solucion_inicial: List[int],
    max_evaluaciones: int = 10000,
    operador_vecindario: Callable = operador_intercambio
) -> Dict[str, Any]:
    """
    Algoritmo de Búsqueda Local básica
    
    Args:
        problema: Instancia del problema
        solucion_inicial: Solución inicial
        max_evaluaciones: Número máximo de evaluaciones
        operador_vecindario: Función para generar vecinos
        
    Returns:
        Diccionario con resultados del algoritmo
    """
    solucion_actual = solucion_inicial.copy()
    coste_actual = problema.evaluar_ruta(solucion_actual)
    mejor_solucion = solucion_actual.copy()
    mejor_coste = coste_actual
    
    evaluaciones = 0
    mejoro = True
    
    while mejoro and evaluaciones < max_evaluaciones:
        mejoro = False
        
        # Explorar vecindario hasta encontrar mejora
        for _ in range(len(solucion_actual) * 2):  # Número arbitrario de intentos
            if evaluaciones >= max_evaluaciones:
                break
                
            vecino = operador_vecindario(solucion_actual)
            coste_vecino = problema.evaluar_ruta(vecino)
            evaluaciones += 1
            
            # Si hay mejora, actualizar solución
            if coste_vecino < coste_actual:
                solucion_actual = vecino
                coste_actual = coste_vecino
                mejoro = True
                
                # Actualizar mejor solución
                if coste_actual < mejor_coste:
                    mejor_solucion = solucion_actual.copy()
                    mejor_coste = coste_actual
                break
    
    return {
        "mejor_solucion": mejor_solucion,
        "mejor_coste": mejor_coste,
        "evaluaciones": evaluaciones
    }

def busqueda_multiarranque_basica(
    problema: ProblemaRutasReparto,
    num_arranques: int = 10,
    max_evaluaciones_por_arranque: int = 10000,
    operador_vecindario: Callable = operador_intercambio
) -> Dict[str, Any]:
    """
    Búsqueda Multiarranque Básica
    
    Args:
        problema: Instancia del problema
        num_arranques: Número de soluciones iniciales
        max_evaluaciones_por_arranque: Número máximo de evaluaciones por arranque
        operador_vecindario: Función para generar vecinos
        
    Returns:
        Diccionario con resultados del algoritmo
    """
    mejor_solucion_global = None
    mejor_coste_global = float('inf')
    evaluaciones_totales = 0
    resultados_arranques = []
    
    print(f"Iniciando Búsqueda Multiarranque Básica con {num_arranques} arranques:")
    
    # Generar y optimizar múltiples soluciones iniciales
    for i in range(num_arranques):
        solucion_inicial = problema.generar_solucion_aleatoria()
        coste_inicial = problema.evaluar_ruta(solucion_inicial)
        
        resultado = busqueda_local_basica(
            problema=problema,
            solucion_inicial=solucion_inicial,
            max_evaluaciones=max_evaluaciones_por_arranque,
            operador_vecindario=operador_vecindario
        )
        
        resultados_arranques.append({
            "arranque_num": i + 1,
            "solucion_inicial": solucion_inicial,
            "coste_inicial": coste_inicial,
            "mejor_solucion": resultado["mejor_solucion"],
            "mejor_coste": resultado["mejor_coste"],
            "evaluaciones": resultado["evaluaciones"]
        })
        
        evaluaciones_totales += resultado["evaluaciones"]
        
        # Actualizar mejor solución global
        if resultado["mejor_coste"] < mejor_coste_global:
            mejor_solucion_global = resultado["mejor_solucion"].copy()
            mejor_coste_global = resultado["mejor_coste"]
        
        print(f"  Arranque {i+1}: Inicial={coste_inicial:.2f}, Final={resultado['mejor_coste']:.2f}, Eval={resultado['evaluaciones']}")
    
    print(f"Búsqueda Multiarranque finalizada:")
    print(f"  - Evaluaciones totales: {evaluaciones_totales}")
    print(f"  - Mejor coste global: {mejor_coste_global:.2f}")
    
    return {
        "mejor_solucion": mejor_solucion_global,
        "mejor_coste": mejor_coste_global,
        "evaluaciones_totales": evaluaciones_totales,
        "resultados_arranques": resultados_arranques
    }

# ---------------------------------
# Comparación experimental
# ---------------------------------

def comparar_algoritmos(problema: ProblemaRutasReparto, num_clientes: int = 20, semilla: int = 42) -> None:
    """
    Comparar los algoritmos ES y BMB en el problema de rutas
    
    Args:
        problema: Instancia del problema
        num_clientes: Número de clientes
        semilla: Semilla para reproducibilidad
    """
    # Configuramos semillas para reproducibilidad
    np.random.seed(semilla)
    random.seed(semilla)
    
    print("="*60)
    print("COMPARACIÓN DE ALGORITMOS DE OPTIMIZACIÓN DE RUTAS")
    print("="*60)
    
    # Ejecutamos Enfriamiento Simulado
    print("\n1. ENFRIAMIENTO SIMULADO (ES)")
    print("-" * 40)
    inicio = time.time()
    resultado_es = enfriamiento_simulado(problema)
    tiempo_es = time.time() - inicio
    
    # Ejecutamos Búsqueda Multiarranque Básica
    print("\n2. BÚSQUEDA MULTIARRANQUE BÁSICA (BMB)")
    print("-" * 40)
    inicio = time.time()
    resultado_bmb = busqueda_multiarranque_basica(problema)
    tiempo_bmb = time.time() - inicio
    
    # Mostramos resultados comparativos
    print("\n3. RESULTADOS COMPARATIVOS")
    print("-" * 40)
    
    print(f"\nEnfriamiento Simulado (ES):")
    print(f"  - Distancia total: {resultado_es['mejor_coste']:.2f}")
    print(f"  - Evaluaciones: {resultado_es['evaluaciones']}")
    print(f"  - Tiempo de ejecución: {tiempo_es:.2f} segundos")
    
    print(f"\nBúsqueda Multiarranque Básica (BMB):")
    print(f"  - Distancia total: {resultado_bmb['mejor_coste']:.2f}")
    print(f"  - Evaluaciones: {resultado_bmb['evaluaciones_totales']}")
    print(f"  - Tiempo de ejecución: {tiempo_bmb:.2f} segundos")
    
    # Determinar ganador
    if resultado_es['mejor_coste'] < resultado_bmb['mejor_coste']:
        ganador = "Enfriamiento Simulado"
        diferencia = resultado_bmb['mejor_coste'] - resultado_es['mejor_coste']
    else:
        ganador = "Búsqueda Multiarranque Básica"
        diferencia = resultado_es['mejor_coste'] - resultado_bmb['mejor_coste']
    
    print(f"\n🏆 GANADOR: {ganador}")
    print(f"   Diferencia: {diferencia:.2f} unidades de distancia")
    
    # Visualizamos las rutas
    print("\n4. VISUALIZACIÓN DE RESULTADOS")
    print("-" * 40)
    problema.graficar_ruta(resultado_es["mejor_solucion"], "Ruta óptima (Enfriamiento Simulado)")
    problema.graficar_ruta(resultado_bmb["mejor_solucion"], "Ruta óptima (Búsqueda Multiarranque)")
    
    # Gráfico de convergencia para ES
    if resultado_es["historial"]:
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Convergencia ES
        plt.subplot(1, 2, 1)
        evaluaciones, costes = zip(*resultado_es["historial"])
        plt.plot(evaluaciones, costes, 'b-', linewidth=2)
        plt.xlabel("Evaluaciones")
        plt.ylabel("Distancia total")
        plt.title("Convergencia del Enfriamiento Simulado")
        plt.grid(True)
          # Subplot 2: Comparativa BMB
        plt.subplot(1, 2, 2)
        arranques = [r["arranque_num"] for r in resultado_bmb["resultados_arranques"]]
        costes_iniciales = [r["coste_inicial"] for r in resultado_bmb["resultados_arranques"]]
        costes_finales = [r["mejor_coste"] for r in resultado_bmb["resultados_arranques"]]
        
        x = np.arange(len(arranques))
        width = 0.35
        
        plt.bar(x - width/2, costes_iniciales, width, alpha=0.7, color='lightblue', label='Coste inicial')
        plt.bar(x + width/2, costes_finales, width, alpha=0.7, color='darkblue', label='Coste final')
        plt.xlabel("Número de arranque")
        plt.ylabel("Distancia total")
        plt.title("Comparativa de arranques en BMB")
        plt.xticks(x, arranques)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("convergencia_algoritmos.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Gráfico de convergencia guardado: convergencia_algoritmos.png")
    
    print(f"\n✅ Gráficos guardados exitosamente")
    print("="*60)

# ---------------------------------
# Análisis de operadores de vecindario
# ---------------------------------

def comparar_operadores(problema: ProblemaRutasReparto, semilla: int = 42) -> None:
    """
    Comparar el rendimiento de los diferentes operadores de vecindario
    """
    print("\n5. ANÁLISIS DE OPERADORES DE VECINDARIO")
    print("-" * 40)
    
    operadores = {
        "Intercambio (Swap)": operador_intercambio,
        "Inserción (Shift)": operador_insercion
    }
    
    resultados_operadores = {}
    
    for nombre, operador in operadores.items():
        print(f"\nProbando operador: {nombre}")
        
        # Probar con ES
        np.random.seed(semilla)
        random.seed(semilla)
        resultado_es = enfriamiento_simulado(problema, operador_vecindario=operador, max_evaluaciones=50000)
        
        # Probar con BMB
        np.random.seed(semilla)
        random.seed(semilla)
        resultado_bmb = busqueda_multiarranque_basica(problema, operador_vecindario=operador, num_arranques=5)
        
        resultados_operadores[nombre] = {
            "es": resultado_es,
            "bmb": resultado_bmb
        }
        
        print(f"  ES: {resultado_es['mejor_coste']:.2f}")
        print(f"  BMB: {resultado_bmb['mejor_coste']:.2f}")
        
        # Visualizar mejores rutas por operador
        problema.graficar_ruta(resultado_es["mejor_solucion"], f"Ruta óptima (ES + {nombre})")
        problema.graficar_ruta(resultado_bmb["mejor_solucion"], f"Ruta óptima (BMB + {nombre})")
    
    return resultados_operadores

# ---------------------------------
# Programa principal
# ---------------------------------

def main():
    """Función principal para ejecutar todos los experimentos"""
    # Configuración del problema
    num_clientes = 20
    semilla = 42
    
    print("MEMORIA FINAL - OPTIMIZACIÓN DE RUTAS DE REPARTO")
    print("Implementación de ES y BMB con operadores de vecindario")
    print("="*60)
    
    # Crear instancia del problema
    problema = ProblemaRutasReparto(num_clientes=num_clientes, semilla=semilla)
    
    print(f"Problema creado:")
    print(f"  - Número de clientes: {num_clientes}")
    print(f"  - Semilla: {semilla}")
    print(f"  - Coordenadas del almacén: ({problema.coordenadas[0, 0]:.2f}, {problema.coordenadas[0, 1]:.2f})")
    
    # Comparar algoritmos principales
    comparar_algoritmos(problema, num_clientes, semilla)
    
    # Comparar operadores de vecindario
    comparar_operadores(problema, semilla)
    
    print("\n🎉 Todos los experimentos completados exitosamente!")
    print("Revisa los archivos .png generados para ver las visualizaciones.")

if __name__ == "__main__":
    main()