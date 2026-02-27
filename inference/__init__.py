"""
inference/
----------
Módulo de inferencia probabilística sobre el Factor Graph.
Implementa consultas marginales y condicionales mediante
el algoritmo Belief Propagation.
"""

from inference.engine import (
    crear_motor_inferencia,
    consultar_marginal,
    consultar_con_evidencia,
    ejecutar_inferencias_demo
)

__all__ = [
    "crear_motor_inferencia",
    "consultar_marginal",
    "consultar_con_evidencia",
    "ejecutar_inferencias_demo"
]