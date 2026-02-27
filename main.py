"""
main.py
-------
Punto de entrada principal del proyecto Factor Graph.

Ejecuta de forma secuencial:
  1. Construcción del Factor Graph manual.
  2. Inferencia con Belief Propagation.
  3. Aprendizaje de estructura desde datos sintéticos.
  4. Visualización del grafo manual y comparación con el aprendido.

Uso:
    python main.py
"""

import os
from model import construir_factor_graph
from inference import crear_motor_inferencia, ejecutar_inferencias_demo
from learning import ejecutar_aprendizaje_demo
from visualization import dibujar_factor_graph, dibujar_comparacion


def main():
    print("=" * 60)
    print("   PROYECTO: FACTOR GRAPH CON PGMPY")
    print("=" * 60)

    # ----------------------------------------------------------
    # PASO 1: Construir el Factor Graph manualmente
    # ----------------------------------------------------------
    print("\nPASO 1: Construcción del Factor Graph")
    fg_manual = construir_factor_graph()

    # ----------------------------------------------------------
    # PASO 2: Inferencia mediante Belief Propagation
    # ----------------------------------------------------------
    print("\nPASO 2: Inferencia")
    bp = crear_motor_inferencia(fg_manual)
    ejecutar_inferencias_demo(bp)

    # ----------------------------------------------------------
    # PASO 3: Aprender estructura desde datos sintéticos
    # ----------------------------------------------------------
    print("\nPASO 3: Aprendizaje de estructura desde datos")
    fg_aprendido = ejecutar_aprendizaje_demo()

    # ----------------------------------------------------------
    # PASO 4: Visualización
    # ----------------------------------------------------------
    print("\nPASO 4: Visualización")

    # Crear carpeta de salida si no existe
    os.makedirs("output", exist_ok=True)

    # Dibujar el Factor Graph manual
    dibujar_factor_graph(
        fg_manual,
        titulo="Factor Graph Manual",
        guardar_en="output/factor_graph_manual.png"
    )

    # Dibujar el Factor Graph aprendido
    dibujar_factor_graph(
        fg_aprendido,
        titulo="Factor Graph Aprendido desde Datos",
        guardar_en="output/factor_graph_aprendido.png"
    )

    # Comparación lado a lado
    dibujar_comparacion(
        fg_manual,
        fg_aprendido,
        guardar_en="output/comparacion.png"
    )

    print("\n" + "=" * 60)
    print("EJECUCIÓN COMPLETADA")
    print("Imágenes guardadas en: ./output/")
    print("=" * 60)


if __name__ == "__main__":
    main()