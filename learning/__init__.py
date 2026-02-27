"""
learning/
---------
Módulo de aprendizaje automático de estructura y parámetros
del Factor Graph a partir de datos tabulares.
"""

from learning.structure_learner import (
    generar_datos_sinteticos,
    aprender_estructura,
    estimar_parametros,
    bayesian_a_factor_graph,
    ejecutar_aprendizaje_demo
)

__all__ = [
    "generar_datos_sinteticos",
    "aprender_estructura",
    "estimar_parametros",
    "bayesian_a_factor_graph",
    "ejecutar_aprendizaje_demo"
]